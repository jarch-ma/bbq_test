"""LIT online adaptation losses, training and prediction for the SAM 3 decoder."""

import copy
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NO_OBJ_SCORE = -1024

def set_seeds(seed_value, max_epochs=10, dist_rank=0):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    # Since in the pytorch sampler, we increment the seed by 1 for every epoch.

    seed_value = (seed_value + dist_rank) * max_epochs
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects

Sample = Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
class VOSLoRADataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch: List[Sample]):
    return {
        "frame_idx": batch[0][0],
        "pix":       batch[0][1],
        "pe":        batch[0][2],
        "sp_emb":    batch[0][3],
        "de_emb":    batch[0][4],
        "hi_res":    batch[0][5],
        "gt_mask":   batch[0][6],
        "original_gt_mask": batch[0][7],
    }

loss_weight_dict = {
    'loss_mask': 20,
    'loss_dice': 1,
}

def loss_fn(src, target, weight_dict=loss_weight_dict, loss_on_multimask=False):
    if loss_on_multimask:
        target = target.unsqueeze(1)
        target = target.expand_as(src)
    loss_mask = sigmoid_focal_loss(src, target, num_objects=1, loss_on_multimask=loss_on_multimask)
    loss_dice = dice_loss(src, target, num_objects=1, loss_on_multimask=loss_on_multimask)
    if loss_on_multimask:
        # take the mask indices with the smallest focal + dice loss for back propagation
        loss_combo = (
            loss_mask * weight_dict["loss_mask"]
            + loss_dice * weight_dict["loss_dice"]
        )
        best_loss_inds = torch.argmin(loss_combo, dim=-1)
        batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
        loss_mask = loss_mask[batch_inds, best_loss_inds].unsqueeze(1)
        loss_dice = loss_dice[batch_inds, best_loss_inds].unsqueeze(1)
        # # calculate the iou prediction and slot losses only in the index
        # # with the minimum loss for each mask (to be consistent w/ SAM)
        # if self.supervise_all_iou:
        #     loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
        # else:
        #     loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
    loss = weight_dict['loss_mask'] * loss_mask + weight_dict['loss_dice'] * loss_dice
    return loss

def get_iou(obj_mask, obj_gt, void_pixel=255):
    obj_void = obj_gt == void_pixel
    obj_void = ~obj_void
    obj_gt = (obj_gt > 0) & (obj_gt != void_pixel)
    intersection = (obj_mask * obj_gt * obj_void).sum()
    pixel_sum = (obj_mask * obj_void).sum() + (obj_gt * obj_void).sum()
    # handle edge cases without resorting to epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        assert intersection == 0
        return 1
    return intersection / (pixel_sum - intersection)

set_seeds(123, 10)

def train_one_epoch(model, loader, optim, loss_fn, device, multimask_output=False):
    model.train()
    total_loss = 0.0
    model = model.to(device)
    loss_list = []
    for batch in loader:
        pix = batch["pix"].detach()
        pe = batch["pe"].detach()
        sp = batch["sp_emb"].detach()
        de = batch["de_emb"].detach()
        hi = batch["hi_res"]
        gt = batch["gt_mask"].detach()

        optim.zero_grad()
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits_lora,
        ) = model(
            image_embeddings=pix,
            image_pe=pe,
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=de,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=hi,
        )
        low_res_multimasks = low_res_multimasks.float()
        is_obj_appearing = object_score_logits_lora > 0
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE
        )
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(gt.shape[2], gt.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        high_res_multimasks = high_res_multimasks.squeeze(1)
        gt = gt.squeeze(1)
        loss = loss_fn(high_res_multimasks, gt, loss_on_multimask=multimask_output)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        loss_list.append(loss.item())
    return total_loss / len(loader.dataset)

def evaluate(loader, mask_decoder_lora):
    total_iou = 0.0
    was_training = mask_decoder_lora.training
    mask_decoder_lora.eval()
    for batch in loader:
        pix = batch["pix"].detach()
        pe = batch["pe"].detach()
        sp = batch["sp_emb"].detach()
        de = batch["de_emb"].detach()
        hi = batch["hi_res"]
        gt = batch["gt_mask"].detach()
        original_gt_mask = batch['original_gt_mask']
        H, W = original_gt_mask.shape[:2]
        with torch.no_grad():
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits_lora,
            ) = mask_decoder_lora(
                pix,
                pe,
                sp,
                de,
                False,
                False,
                hi,
            )
            is_obj_appearing = object_score_logits_lora > 0
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE
            )
            lora_predicted_mask = F.interpolate(
                low_res_multimasks,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            lora_predicted_mask = lora_predicted_mask.squeeze(0).squeeze(0)
            lora_predicted_mask = (lora_predicted_mask > 0).cpu().numpy().astype(np.uint8)
            lora_predicted_mask = lora_predicted_mask.reshape(H, W)
            iou = get_iou(lora_predicted_mask, original_gt_mask)
            total_iou += iou
    if was_training:
        mask_decoder_lora.train()
    return total_iou / len(loader.dataset)



def train_with_random_split_each_epoch(
    model,
    full_dataset,
    optimizer,
    loss_fn,
    device,
    max_epochs=100,
    patience=10,
    batch_size=1,
):
    best_val_iou = -1
    epochs_no_improve = 0
    dataset_loader = DataLoader(VOSLoRADataset(full_dataset), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, dataset_loader, optimizer, loss_fn, device)
        val_iou = evaluate(dataset_loader, model)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and best_val_iou > 0.6 or \
            (epoch == 10 and best_val_iou == 0):
            break
    model = best_model
    return model, best_val_iou

def predict(
    model,
    pix_feat_with_mem,
    image_pe,
    sparse_embeddings,
    dense_embeddings,
    high_res_features,
    image_size,
    original_gt_mask=None,
):
    model.eval()
    with torch.no_grad():
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits_lora,
        ) = model(
            image_embeddings=pix_feat_with_mem,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
    # convert masks from possibly bfloat16 (or float16) to float32
    # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
    low_res_multimasks = low_res_multimasks.float()
    is_obj_appearing = object_score_logits_lora > 0
    low_res_multimasks = torch.where(
        is_obj_appearing[:, None, None],
        low_res_multimasks,
        NO_OBJ_SCORE
    )
    high_res_multimasks = F.interpolate(
        low_res_multimasks,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    high_res_multimasks_ = high_res_multimasks.squeeze(0).squeeze(0)
    high_res_multimasks_ = (high_res_multimasks_ > 0).cpu().numpy().astype(np.uint8)

    low_res_masks_lora, high_res_masks_lora = low_res_multimasks, high_res_multimasks
    H, W = original_gt_mask.shape[:2]
    lora_predicted_mask_score = F.interpolate(
        low_res_masks_lora,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    lora_predicted_mask = lora_predicted_mask_score.squeeze(0).squeeze(0)
    lora_predicted_mask = (lora_predicted_mask > 0).cpu().numpy().astype(np.uint8)
    lora_predicted_mask = lora_predicted_mask.reshape(H, W)
    lora_iou = get_iou(lora_predicted_mask, original_gt_mask)
    lora_iou_predicted = ious[0].item()
    
    output = (
        low_res_masks_lora,
        high_res_masks_lora,
        object_score_logits_lora,
        # sam_output_tokens,
        # lora_iou_predicted,
        lora_iou,
        lora_predicted_mask_score,
        lora_predicted_mask,
        # lora_iou_predicted
    )
    return output
