"""Adapt Meta's SAM 3 tracker to the API used by LIT-SAM.

The upstream SAM 3 video tracker is deliberately kept unmodified.  This module
loads only the tracker and visual-backbone weights from a full SAM 3 checkpoint,
preserves the existing LIT evaluation API, and implements the LIT online
LoRA hooks around the SAM-style mask decoder.
"""

from __future__ import annotations

import copy
import gc
import math
import os
import weakref
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import (
    get_iou,
    loss_fn,
    predict,
    train_with_random_split_each_epoch,
)


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual.

    Unlike the original LIT helper, this keeps an existing projection bias. SAM
    3's q/k/v projections have biases, so dropping them would alter the decoder
    output even before the first LoRA update.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

        factory_kwargs = {"device": weight.device, "dtype": weight.dtype}
        self.lora_A = nn.Linear(in_features, rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(rank, out_features, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = self.lora_B(self.lora_A(self.dropout(x)))
        return base + (self.alpha / self.rank) * delta


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


def _clone_for_training(value: Any) -> Any:
    """Turn inference tensors into ordinary detached tensors for autograd."""
    if isinstance(value, torch.Tensor):
        with torch.inference_mode(False):
            return value.detach().clone()
    if isinstance(value, list):
        return [_clone_for_training(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_for_training(item) for item in value)
    return value


def _scores_to_list(scores: Any, object_count: int) -> list[float]:
    if scores is None:
        return [float("nan")] * object_count
    if isinstance(scores, torch.Tensor):
        values = scores.detach().float().reshape(-1).cpu().tolist()
    elif isinstance(scores, (list, tuple)):
        values = [float(value) for value in scores]
    else:
        values = [float(scores)]
    if not values:
        return [float("nan")] * object_count
    if len(values) == 1 and object_count > 1:
        values *= object_count
    return values[:object_count]


class LITSam3VideoPredictor:
    """LIT-compatible facade around ``Sam3TrackerPredictor``."""

    def __init__(self, predictor: nn.Module) -> None:
        self._predictor = predictor
        self.LIT_LoRA_mode = False
        self._active_frame_idx: int | None = None
        self._pending_lora_features: dict[str, Any] | None = None
        self._decoder_hook_handle = None
        self._install_decoder_hook()
        self.reset_lora()

    def __getattr__(self, name: str) -> Any:
        predictor = object.__getattribute__(self, "_predictor")
        return getattr(predictor, name)

    @property
    def sam_mask_decoder(self) -> nn.Module:
        return self._predictor.sam_mask_decoder

    @property
    def sam_prompt_encoder(self) -> nn.Module:
        return self._predictor.sam_prompt_encoder

    def _install_decoder_hook(self) -> None:
        owner_ref = weakref.ref(self)

        def capture_decoder_inputs(module, args, kwargs, output):
            owner = owner_ref()
            if owner is None or not owner.LIT_LoRA_mode:
                return
            if module is not owner._predictor.sam_mask_decoder:
                return
            owner._pending_lora_features = {
                "pix_feat_with_mem": _detach_tree(kwargs["image_embeddings"]),
                "image_pe": _detach_tree(kwargs["image_pe"]),
                "high_res_features": _detach_tree(kwargs.get("high_res_features")),
            }
            if owner._active_frame_idx is not None:
                owner._commit_lora_features(owner._active_frame_idx)

        self._decoder_hook_handle = self.sam_mask_decoder.register_forward_hook(
            capture_decoder_inputs, with_kwargs=True
        )

    def _commit_lora_features(self, frame_idx: int) -> None:
        captured = self._pending_lora_features
        if captured is None:
            return
        self.temp_feat_for_lora.update(captured)
        self.temp_feat_for_lora["frame_idx"] = frame_idx
        self._pending_lora_features = None

    def init_state(self, *args, **kwargs):
        video_path = kwargs.get("video_path")
        if args or not isinstance(video_path, str) or not os.path.isdir(video_path):
            return self._predictor.init_state(*args, **kwargs)

        # SAM 3's bundled JPEG loader requires purely numeric stems. LIT datasets
        # also use names such as ``frame00000.jpg`` (and occasionally PNG/BMP),
        # which the project's loader already handles. Keep SAM 3's normalization.
        from .video_loading import load_video_frames

        offload_video_to_cpu = kwargs.get("offload_video_to_cpu", False)
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=(0.5, 0.5, 0.5),
            img_std=(0.5, 0.5, 0.5),
            async_loading_frames=kwargs.get("async_loading_frames", False),
            compute_device=self.device,
        )
        state_kwargs = dict(kwargs)
        state_kwargs.pop("video_path", None)
        state_kwargs.pop("video_height", None)
        state_kwargs.pop("video_width", None)
        state_kwargs.pop("num_frames", None)
        inference_state = self._predictor.init_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=len(images),
            video_path=None,
            **state_kwargs,
        )
        inference_state["images"] = images
        return inference_state

    def reset_state(self, inference_state) -> None:
        self._predictor.clear_all_points_in_video(inference_state)
        self._pending_lora_features = None

    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        run_mem_encoder=True,
        **kwargs,
    ):
        del run_mem_encoder  # SAM 3 encodes finalized prompts during preflight.
        mask = torch.as_tensor(mask, dtype=torch.bool)
        self._active_frame_idx = frame_idx
        self._pending_lora_features = None
        try:
            out_frame_idx, obj_ids, _, video_res_masks = self._predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
                **kwargs,
            )
        finally:
            self._commit_lora_features(frame_idx)
            self._active_frame_idx = None
        return out_frame_idx, obj_ids, video_res_masks

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        **kwargs,
    ):
        # LIT accepts original-image pixel coordinates by default. SAM 3's
        # ``rel_coordinates`` path expects coordinates in [0, 1].
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32)
            if normalize_coords:
                scale = points.new_tensor(
                    [inference_state["video_width"], inference_state["video_height"]]
                )
                points = points / scale
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float32)
            if normalize_coords:
                original_shape = box.shape
                scale = box.new_tensor(
                    [inference_state["video_width"], inference_state["video_height"]]
                )
                box = (box.reshape(-1, 2) / scale).reshape(original_shape)

        self._active_frame_idx = frame_idx
        self._pending_lora_features = None
        try:
            out_frame_idx, obj_ids, _, video_res_masks = (
                self._predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=clear_old_points,
                    rel_coordinates=normalize_coords,
                    box=box,
                    **kwargs,
                )
            )
        finally:
            self._commit_lora_features(frame_idx)
            self._active_frame_idx = None
        return out_frame_idx, obj_ids, video_res_masks

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        **kwargs,
    ) -> Iterable[tuple[int, list[int], torch.Tensor, list[float]]]:
        kwargs.pop("propagate_preflight", None)
        iterator = self._predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            propagate_preflight=True,
            **kwargs,
        )
        while True:
            self._pending_lora_features = None
            try:
                frame_idx, obj_ids, _, video_res_masks, object_scores = next(iterator)
            except StopIteration:
                break
            self._commit_lora_features(frame_idx)
            current_out = None
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                current_out = inference_state["output_dict"][storage_key].get(frame_idx)
                if current_out is not None:
                    break
            predicted_scores = None if current_out is None else current_out.get("iou_score")
            if predicted_scores is None:
                predicted_scores = object_scores
            yield (
                frame_idx,
                obj_ids,
                video_res_masks,
                _scores_to_list(predicted_scores, len(obj_ids)),
            )

    def propagate_in_video_preflight(self, inference_state, run_mem_encoder=True):
        return self._predictor.propagate_in_video_preflight(
            inference_state, run_mem_encoder=run_mem_encoder
        )

    def correct_by_iou(
        self,
        inference_state,
        frame_idx,
        obj_id,
        cur_iou,
        gt_mask,
        pred_mask_tensor,
        correct_threshold,
        max_num_clicks=3,
    ):
        from sam3.model.sam3_tracker_utils import get_next_point

        height, width = gt_mask.shape
        input_gt_mask = (gt_mask > 0) & (gt_mask != 255)
        gt_mask_tensor = (
            torch.from_numpy(input_gt_mask)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(pred_mask_tensor.device)
        )
        cur_num_clicks = 0
        video_res_masks = pred_mask_tensor.float()

        while cur_iou < correct_threshold and cur_num_clicks < max_num_clicks:
            new_points, new_labels = get_next_point(
                gt_masks=gt_mask_tensor,
                pred_masks=pred_mask_tensor,
                method="center",
            )
            _, _, video_res_masks = self.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=new_points,
                labels=new_labels,
            )
            pred_mask_tensor = video_res_masks > 0
            point_mask = (video_res_masks[0] > 0).cpu().numpy().reshape(height, width)
            cur_iou = get_iou(point_mask, gt_mask)
            cur_num_clicks += 1

        if cur_iou < correct_threshold:
            _, _, video_res_masks = self.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=input_gt_mask,
                run_mem_encoder=True,
            )
            correct_type = "mask"
            cur_iou = 1.0
        else:
            correct_type = "point"
        self.propagate_in_video_preflight(inference_state)

        return {
            "correct_type": correct_type,
            "correct_num_clicks": cur_num_clicks,
            "correct_iou": cur_iou,
            "correct_res_mask": video_res_masks,
        }

    def reset_lora(self) -> None:
        self.temp_feat_for_lora = {
            "frame_idx": -1,
            "current_vision_feats": None,
            "feat_sizes": None,
            "high_res_features": None,
            "pix_feat_with_mem": None,
            "image_pe": None,
        }
        self.sparse_embeddings_for_lora = None
        self.dense_embeddings_for_lora = None
        self.correction_buff = []
        self.trained_lora = False
        self.LIT_lora = None

    def clone_mask_decoder_for_lora(self) -> nn.Module:
        cloned = copy.deepcopy(self.sam_mask_decoder)
        # ``deepcopy`` preserves hooks. The copied decoder must not update the
        # tracker's feature cache during LoRA optimization or prediction.
        cloned._forward_hooks.clear()
        cloned._forward_pre_hooks.clear()
        return cloned

    @staticmethod
    def freeze_non_lora(model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("lora_A" in name or "lora_B" in name)

    def convert_to_lora(
        self,
        module: nn.Module,
        target_module_names=("q_proj", "k_proj", "v_proj"),
        r=4,
        alpha=4,
        dropout=0.1,
    ) -> None:
        for name, child in list(module.named_children()):
            if name in target_module_names and isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    LoRALinear(
                        weight=child.weight,
                        bias=child.bias,
                        rank=r,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
            else:
                self.convert_to_lora(
                    child,
                    target_module_names=target_module_names,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

    def _make_train_sample(self, gt_mask):
        pix_feat_with_mem = self.temp_feat_for_lora["pix_feat_with_mem"]
        high_res_features = self.temp_feat_for_lora["high_res_features"]
        image_pe = self.temp_feat_for_lora["image_pe"]
        if pix_feat_with_mem is None or image_pe is None:
            return None

        pix_feat_with_mem = _clone_for_training(pix_feat_with_mem)
        high_res_features = _clone_for_training(high_res_features)
        image_pe = _clone_for_training(image_pe)
        device = pix_feat_with_mem.device
        batch_size = pix_feat_with_mem.size(0)
        with torch.inference_mode(False), torch.no_grad():
            point_coords = torch.zeros(batch_size, 1, 2, device=device)
            point_labels = -torch.ones(
                batch_size, 1, dtype=torch.int32, device=device
            )
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                points=(point_coords, point_labels), boxes=None, masks=None
            )
            sparse_embeddings = sparse_embeddings.detach().clone()
            dense_embeddings = dense_embeddings.detach().clone()
            gt_mask_resized = torch.as_tensor(
                gt_mask, dtype=torch.bool, device=device
            ).float()[None, None]
            gt_mask_resized = F.interpolate(
                gt_mask_resized,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
            gt_mask_resized = (gt_mask_resized >= 0.5).float()

        self.sparse_embeddings_for_lora = sparse_embeddings
        self.dense_embeddings_for_lora = dense_embeddings
        return (
            self.temp_feat_for_lora["frame_idx"],
            pix_feat_with_mem,
            image_pe,
            sparse_embeddings,
            dense_embeddings,
            high_res_features,
            gt_mask_resized,
            np.asarray(gt_mask),
        )

    def prepare_train_data(self, gt_mask):
        sample = self._make_train_sample(gt_mask)
        return (sample is not None), sample

    def train_lora(self, model, gt_mask, training_epoch=300, mode="init") -> bool:
        del mode  # Both initial training and fine-tuning retain the best model.
        sample = self._make_train_sample(gt_mask)
        if sample is None:
            return False
        device = sample[1].device
        # Online evaluation is driven by an inference-mode video generator. In
        # particular, a second correction can otherwise inherit a disabled-grad
        # context after LoRA prediction and produce a loss without a grad_fn.
        with torch.inference_mode(False), torch.enable_grad():
            optimizer = torch.optim.AdamW(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=1e-4,
                weight_decay=0,
            )
            # Keep optimizing the same decoder instance, matching the original
            # LIT-SAM behavior. The helper's validation snapshot is only used for
            # early stopping; replacing the live decoder with that deep copy makes
            # repeated online fine-tuning lose its active autograd path.
            train_with_random_split_each_epoch(
                model=model,
                full_dataset=[sample],
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                max_epochs=training_epoch,
            )
        self.LIT_lora = model
        self.trained_lora = True
        return True

    def lora_predict(
        self,
        inference_state,
        frame_idx,
        gt_mask,
        object_id,
        correction_threshold=0.5,
    ):
        if self.LIT_lora is None or self.temp_feat_for_lora["pix_feat_with_mem"] is None:
            return False, float("-inf"), None
        image_pe = self.temp_feat_for_lora["image_pe"]
        with torch.no_grad():
            (
                low_res_masks,
                _,
                _,
                lora_iou,
                predicted_mask_score,
                predicted_mask,
            ) = predict(
                model=self.LIT_lora,
                pix_feat_with_mem=self.temp_feat_for_lora["pix_feat_with_mem"],
                image_pe=image_pe,
                sparse_embeddings=self.sparse_embeddings_for_lora,
                dense_embeddings=self.dense_embeddings_for_lora,
                high_res_features=self.temp_feat_for_lora["high_res_features"],
                image_size=self.image_size,
                original_gt_mask=gt_mask,
            )

        if self.fill_hole_area > 0:
            from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores

            low_res_masks = fill_holes_in_mask_scores(
                low_res_masks, self.fill_hole_area
            )
            _, predicted_mask_score = self._predictor._get_orig_video_res_output(
                inference_state, low_res_masks
            )
            height, width = gt_mask.shape[:2]
            predicted_mask = (
                (predicted_mask_score.squeeze(0).squeeze(0) > 0)
                .cpu()
                .numpy()
                .astype(np.uint8)
                .reshape(height, width)
            )
            lora_iou = get_iou(predicted_mask, gt_mask)

        successful = lora_iou >= correction_threshold
        if successful:
            self.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                mask=predicted_mask,
                run_mem_encoder=False,
            )
            self.propagate_in_video_preflight(inference_state)
        return successful, lora_iou, predicted_mask_score


def _resolve_checkpoint(checkpoint_path: str | os.PathLike[str]) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_dir():
        path = path / "sam3.pt"
    if not path.is_file():
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {path}")
    return path


def _tracker_state_from_full_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    tracker_state = {}
    tracker_prefix = "tracker."
    vision_prefix = "detector.backbone.vision_backbone."
    for key, value in checkpoint.items():
        if key.startswith(tracker_prefix):
            tracker_state[key[len(tracker_prefix) :]] = value
        elif key.startswith(vision_prefix):
            tracker_state["backbone.vision_backbone." + key[len(vision_prefix) :]] = value
    return tracker_state


def build_lit_sam3_video_predictor(
    checkpoint_path: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    apply_temporal_disambiguation: bool = True,
    apply_postprocessing: bool = False,
    non_overlap_masks_for_output: bool = True,
) -> LITSam3VideoPredictor:
    """Build a memory-conscious SAM 3 VOS tracker from a full checkpoint.

    Only the tracker (``tracker.*``) and detector visual backbone
    (``detector.backbone.vision_backbone.*``) are materialized. The text encoder
    and open-vocabulary detector are not used by mask-prompt VOS and would add
    substantial memory without changing this evaluation path.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            "The current upstream SAM 3 tracker is CUDA-only; use --device cuda."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, so the SAM 3 tracker cannot be built.")

    try:
        from sam3.model_builder import build_tracker
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "SAM 3 is not importable in this Python environment. Install the official "
            "SAM 3 package before selecting --model_backend sam3."
        ) from error

    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    tracker = build_tracker(
        apply_temporal_disambiguation=apply_temporal_disambiguation,
        with_backbone=True,
        compile_mode=None,
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    tracker_state = _tracker_state_from_full_checkpoint(checkpoint)
    del checkpoint
    gc.collect()

    if not tracker_state:
        raise RuntimeError(
            f"{checkpoint_path} is not a Meta SAM 3 video checkpoint: no tracker weights found."
        )
    incompatible = tracker.load_state_dict(
        tracker_state, strict=False, assign=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys[:8])
        unexpected = ", ".join(incompatible.unexpected_keys[:8])
        raise RuntimeError(
            "SAM 3 checkpoint does not match the installed SAM 3 source. "
            f"Missing keys: [{missing}]; unexpected keys: [{unexpected}]"
        )
    del tracker_state
    gc.collect()

    tracker.add_all_frames_to_correct_as_cond = False
    tracker.non_overlap_masks_for_output = non_overlap_masks_for_output
    tracker.fill_hole_area = 8 if apply_postprocessing else 0
    tracker.eval().to(device)
    return LITSam3VideoPredictor(tracker)
