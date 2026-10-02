# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import argparse
import copy
import gc
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Event, Thread

import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.dataset_config import get_dataset_config
from eval_vos_jf import evaluate_jf
from lit_sam3.annotations import load_object_annotations
from vis_utils import *


@contextmanager
def report_video_peak_memory(video_name, device, sample_interval=0.1):
    """Report this process's sampled RSS and PyTorch CUDA peaks per video.
    Includes the already-loaded model. RSS is sampled every 100 ms by default;
    CUDA allocated/reserved peaks come from the PyTorch allocator, not the
    entire GPU. A forced process kill cannot run the final report.
    """
    memory_stats = {"video_name": video_name, "cpu_rss_peak_gib": None, "gpu_allocated_peak_gib": None, "gpu_reserved_peak_gib": None}
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    stop = Event()
    device = torch.device(device)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def sample_rss():
        nonlocal peak_rss
        while not stop.wait(sample_interval):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        yield memory_stats
    finally:
        stop.set()
        sampler.join()
        peak_rss = max(peak_rss, process.memory_info().rss)
        gib = 1024 ** 3
        memory_stats["cpu_rss_peak_gib"] = peak_rss / gib
        gpu_report = "GPU: N/A"
        if use_cuda:
            allocated = torch.cuda.max_memory_allocated(device) / gib
            reserved = torch.cuda.max_memory_reserved(device) / gib
            memory_stats["gpu_allocated_peak_gib"] = allocated
            memory_stats["gpu_reserved_peak_gib"] = reserved
            gpu_report = f"GPU allocated peak: {allocated:.3f} GiB | GPU reserved peak: {reserved:.3f} GiB"
        print(f"\n[video memory] {video_name} | CPU RSS peak (sampled): {peak_rss / gib:.3f} GiB | {gpu_report}", flush=True)


def get_single_predicted_iou(iou_per_obj):
    """Return one JSON-friendly score from SAM 3 API."""
    if isinstance(iou_per_obj, torch.Tensor):
        values = iou_per_obj.detach().float().reshape(-1).cpu().tolist()
    elif isinstance(iou_per_obj, (list, tuple)):
        values = iou_per_obj
    elif iou_per_obj is None:
        values = []
    else:
        values = [iou_per_obj]
    return float(values[0]) if values else float("nan")
# 后面加的


def create_empty_interaction_stats(video_name=None):
    """
    Statistics of real user interactions.
    One correction = one user-correction event on one object/frame.
    One correction may contain multiple point clicks.
    """
    return {
        "video_name": video_name,
        "user_corrections": 0,
        "user_clicks": 0,
        "point_corrections": 0,
        "mask_corrections": 0,
        "lora_auto_corrections": 0,
    }


def accumulate_interaction_stats(stats, output_prompt_per_object):
    """
    Accumulate interaction statistics from one evaluation pass.
    output_prompt_per_object:
        {
            object_id: {
                frame_idx: {
                    "type": "user_correction" / "lora_corrected",
                    "correct_type": "point" / "mask",
                    "correct_num_clicks": int,
                    ...
                }
            }
        }
    For the non-LoRA branch, old cache entries may not contain "type".
    They are treated as user corrections.
    """
    for object_id, frame_records in output_prompt_per_object.items():
        if not isinstance(frame_records, dict):
            continue
        for frame_idx, record in frame_records.items():
            if not isinstance(record, dict):
                continue
            # Non-LoRA records historically have no "type",
            # and every record there is a genuine user correction.
            event_type = record.get("type", "user_correction")
            if event_type == "user_correction":
                stats["user_corrections"] += 1
                num_clicks = record.get("correct_num_clicks", 0)
                if num_clicks is None:
                    num_clicks = 0
                stats["user_clicks"] += int(num_clicks)
                correct_type = record.get("correct_type", None)
                if correct_type == "point":
                    stats["point_corrections"] += 1
                elif correct_type == "mask":
                    stats["mask_corrections"] += 1
            elif event_type == "lora_corrected":
                # This is NOT a user interaction.
                stats["lora_auto_corrections"] += 1
    return stats
# the PNG palette for DAVIS 2017 dataset


DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    # 255 is the standard ignore/void label, not a trackable object id.
    object_ids = object_ids[(object_ids > 0) & (object_ids != 255)].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False):
    """Load masks from a directory as a dict of per-object masks."""
    sufix = '.bmp' if 'endovis' in input_mask_dir else '.png'
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}{sufix}")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(input_mask_dir, video_name, object_name, f"{frame_name}{sufix}")
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0
    return per_obj_input_mask, input_palette


def save_masks_to_dir(output_mask_dir, video_name, frame_name, per_obj_output_mask, height, width, per_obj_png_file, output_palette):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(output_mask_dir, video_name, f"{frame_name}.png")
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(os.path.join(output_mask_dir, video_name, object_name), exist_ok=True)
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            # output_mask[output_mask > 0] = object_id
            output_mask_path = os.path.join(output_mask_dir, video_name, object_name, f"{frame_name}.png")
            save_ann_png(output_mask_path, output_mask, output_palette)


def get_input_per_obj(base_video_dir, input_mask_dir, video_name, use_all_masks=False, per_obj_png_file=False, track_object_appearing_later_in_video=False):
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [os.path.splitext(p)[0] for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".bmp", ".PNG", ".BMP"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].replace('frame', '').split('_')[-1]))
    # 只检查第一帧有没有 GT mask
    if not track_object_appearing_later_in_video:
        frame_names = frame_names[:1]
    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    sufix = '.bmp' if 'endovis' in base_video_dir else '.png'
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or os.path.exists(os.path.join(input_mask_dir, video_name, f"{name}{sufix}")):  # 如果存在mask就读取
            # input_palette：原来颜色的映射表
            per_obj_input_mask, input_palette = load_masks_from_dir(input_mask_dir=input_mask_dir, video_name=video_name, frame_name=frame_names[idx], per_obj_png_file=per_obj_png_file, allow_missing=True)
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask
    return input_palette, inputs_per_object


def vos_online_one_pass_user_correction(
    predictor,
    inference_state,
    inputs_per_object,
    obj_gt_mask_per_object,
    video_segments,
    object_id,
    score_thresh,
    correct_threshold,
    correction_num_limit,
    max_num_click_per_frame=3,
):
    input_frame_inds = sorted(inputs_per_object[object_id])
    cur_correction_num = 0
    output_gt_iou_per_object = []
    output_pred_iou_per_object = []
    output_prompt_per_object = {}
    finish_process_obj = False
    predictor.reset_state(inference_state)
    for input_frame_idx in input_frame_inds:
        predictor.add_new_mask(inference_state=inference_state, frame_idx=input_frame_idx, obj_id=object_id, mask=inputs_per_object[object_id][input_frame_idx])
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    # run propagation throughout the video and collect the results in a dict
    for out_frame_idx, _, out_mask_logits, iou_per_obj in predictor.propagate_in_video(inference_state, start_frame_idx=min(input_frame_inds), reverse=False):
        video_segments[out_frame_idx][object_id] = (out_mask_logits[0] > 0).cpu().numpy()
        mask = (out_mask_logits[0] > score_thresh).cpu().numpy()
        mask = mask.reshape(height, width)
        output_pred_iou_per_object.append(get_single_predicted_iou(iou_per_obj))
        if obj_gt_mask_per_object[object_id][out_frame_idx] is not None:
            gt_mask = obj_gt_mask_per_object[object_id][out_frame_idx]
            gt_iou = get_iou(mask, gt_mask)
            prev_iou = gt_iou
            input_gt_mask = ((gt_mask > 0) & (gt_mask != 255)).astype(np.uint8)
            if gt_iou < correct_threshold and (correction_num_limit == None or cur_correction_num < correction_num_limit):
                print(f"[user correction] frame: {out_frame_idx}, iou: {gt_iou:.4f}")
                correct_result = predictor.correct_by_iou(
                    inference_state=inference_state,
                    frame_idx=out_frame_idx,
                    obj_id=object_id,
                    cur_iou=gt_iou,
                    gt_mask=input_gt_mask,
                    pred_mask_tensor=out_mask_logits>0,
                    correct_threshold=correct_threshold,
                    max_num_clicks=max_num_click_per_frame,
                )
                gt_iou = correct_result['correct_iou']
                correct_res_mask = correct_result['correct_res_mask']
                video_segments[out_frame_idx][object_id] = (correct_res_mask[0] > 0).cpu().numpy()
                # predictor.propagate_in_video_preflight(inference_state)
                output_prompt_per_object[out_frame_idx] = {
                    'prev_iou': float(prev_iou),
                    'prev_mask': bmask_to_rle(mask.astype(np.bool_)),
                    'type': 'user_correction',
                    'correct_type': correct_result['correct_type'],
                    'correct_num_clicks': correct_result['correct_num_clicks'],
                    }
                cur_correction_num += 1
            output_gt_iou_per_object.append(gt_iou)
    if output_gt_iou_per_object and min(output_gt_iou_per_object) >= correct_threshold:
        finish_process_obj = True
    print(f"[Without LIT-LoRA user correction num]: {cur_correction_num}")
    return output_gt_iou_per_object, output_pred_iou_per_object, output_prompt_per_object, finish_process_obj


def vos_online_one_pass_lora_correction(
    predictor,
    inference_state,
    inputs_per_object,
    obj_gt_mask_per_object,
    video_segments,
    object_id,
    score_thresh,
    correct_threshold,
    correction_num_limit,
    max_num_click_per_frame=3,
    training_epoch=40,
):
    trained_lora = False
    input_frame_inds = sorted(inputs_per_object[object_id])
    output_gt_iou_per_object = []
    output_pred_iou_per_object = [] # 这是SAM自己预测的IOU，不准
    output_prompt_per_object = {}
    finish_process_obj = False
    cur_correction_num = 0
    predictor.reset_state(inference_state)
    predictor.reset_lora()
    for input_frame_idx in input_frame_inds:
        # 输入当前帧GT mask，确定目标对象
        predictor.add_new_mask(inference_state=inference_state, frame_idx=input_frame_idx, obj_id=object_id, mask=inputs_per_object[object_id][input_frame_idx])
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    # run propagation throughout the video and collect the results in a dict
    # 开始预测每一帧的目标对象，给出分割结果
    for out_frame_idx, _, out_mask_logits, iou_per_obj in predictor.propagate_in_video(
        inference_state, #前面的视频状态
        start_frame_idx=min(input_frame_inds), # 从哪一帧开始传播
        reverse=False, # 正向传播
    ):
        video_segments[out_frame_idx][object_id] = (out_mask_logits[0] > 0).cpu().numpy()
        # 预测的mask
        pred_mask = (out_mask_logits[0] > score_thresh).cpu().numpy().reshape(height, width)
        # 这是SAM自己预测的IOU
        output_pred_iou_per_object.append(get_single_predicted_iou(iou_per_obj))
        if obj_gt_mask_per_object[object_id][out_frame_idx] is not None:
            gt_mask = obj_gt_mask_per_object[object_id][out_frame_idx]
            gt_iou = get_iou(pred_mask, gt_mask)
            input_gt_mask = ((gt_mask > 0) & (gt_mask != 255)).astype(np.uint8)
            print(f'frame: {out_frame_idx}, gt_iou: {gt_iou:.4f}')
            if gt_iou < correct_threshold and (correction_num_limit == None or cur_correction_num < correction_num_limit):
                prev_iou = gt_iou
                user_correct_type = None
                user_correct_num_clicks = 0
                correction_type = None
                if not trained_lora:
                    # correct by users
                    print(f"[user correction] frame: {out_frame_idx}, iou: {gt_iou:.4f}")
                    # 预测新的mask
                    correct_result = predictor.correct_by_iou(
                        inference_state=inference_state,
                        frame_idx=out_frame_idx,
                        obj_id=object_id,
                        cur_iou=gt_iou,
                        gt_mask=input_gt_mask,
                        pred_mask_tensor=out_mask_logits>0,
                        correct_threshold=correct_threshold,
                        max_num_clicks=max_num_click_per_frame,
                    )
                    user_correct_type = correct_result['correct_type']
                    user_correct_num_clicks = correct_result['correct_num_clicks']
                    gt_iou = correct_result['correct_iou']
                    correct_res_mask = correct_result['correct_res_mask']
                    video_segments[out_frame_idx][object_id] = (correct_res_mask[0] > 0).cpu().numpy()
                    correction_type = 'user_correction'
                    cur_correction_num += 1
                    # init the LIT_LoRA
                    # 复制sam decoder作备份
                    if hasattr(predictor, "clone_mask_decoder_for_lora"):
                        mask_decoder_lora = predictor.clone_mask_decoder_for_lora()
                    else:
                        mask_decoder_lora = copy.deepcopy(predictor.sam_mask_decoder)
                    # 把transformer转换成Lora形式
                    predictor.convert_to_lora(mask_decoder_lora.transformer)
                    # 除了Lora之外的参数冻结
                    predictor.freeze_non_lora(mask_decoder_lora)
                    # 训练Lora
                    if predictor.train_lora(
                        mask_decoder_lora,
                        gt_mask,
                        training_epoch=training_epoch,
                    ):
                        trained_lora = True
                else:
                    # check whether LoRA prediction is good
                    successful_predict, best_lora_iou, best_lora_predicted_mask_score = predictor.lora_predict(
                        inference_state,
                        out_frame_idx,
                        gt_mask,
                        object_id,
                        correct_threshold
                    )
                    # if LoRA prediction is bad, turn to user corrections
                    if not successful_predict and (correction_num_limit == None or cur_correction_num < correction_num_limit):
                        print(f"[user correction] frame: {out_frame_idx}, iou: {gt_iou:.4f}")
                        correct_result = predictor.correct_by_iou(
                            inference_state=inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=object_id,
                            cur_iou=gt_iou,
                            gt_mask=gt_mask,
                            pred_mask_tensor=out_mask_logits>0,
                            correct_threshold=correct_threshold,
                            max_num_clicks=max_num_click_per_frame,
                        )
                        user_correct_type = correct_result['correct_type']
                        user_correct_num_clicks = correct_result['correct_num_clicks']
                        gt_iou = correct_result['correct_iou']
                        correct_res_mask = correct_result['correct_res_mask']
                        video_segments[out_frame_idx][object_id] = (correct_res_mask[0] > 0).cpu().numpy()
                        correction_type = 'user_correction'
                        cur_correction_num += 1
                        predictor.train_lora(
                            predictor.LIT_lora,
                            gt_mask,
                            training_epoch=training_epoch,
                            mode='finetune',
                        )
                    # accept LIT_LoRA prediction if sucessful
                    elif successful_predict:
                        print(f"[LIT-LoRA correction] frame: {out_frame_idx}, original iou: {gt_iou:.4f}, lora iou: {best_lora_iou:.4f}")
                        correction_type = 'lora_corrected'
                        video_segments[out_frame_idx][object_id] = (best_lora_predicted_mask_score[0] > 0).cpu().numpy()
                output_prompt_per_object[out_frame_idx] = {
                    'prev_iou': float(prev_iou),
                    'prev_mask': bmask_to_rle(pred_mask.astype(np.bool_)),
                    'type': correction_type,
                    'correct_type': user_correct_type,
                    'correct_num_clicks': user_correct_num_clicks,
                }
            output_gt_iou_per_object.append(gt_iou)
    if output_gt_iou_per_object and min(output_gt_iou_per_object) >= correct_threshold:
        finish_process_obj = True
    print(f"[LIT-LoRA user correction num]: {cur_correction_num}")
    return output_gt_iou_per_object, output_pred_iou_per_object, output_prompt_per_object, finish_process_obj


def vos_online_evaluation_pass(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    LIT_LoRA_mode=False,
    max_num_click_per_frame=3, # number of corrections per frame,
    correct_threshold=0.75,
    track_object_appearing_later_in_video=False,
    pass_num=1,
    video_obj_list=None,
    training_epoch=40,
    offload_video_to_cpu=None,
    offload_state_to_cpu=False,
    async_loading_frames=False,
):
    #读取数据集
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png', '.bmp']
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].replace('frame', '').split('_')[-1]))
    gt_mask_dir = os.path.join(input_mask_dir, video_name)
    # Match the original LIT policy unless explicitly overridden by the caller.
    if offload_video_to_cpu is None:
        offload_video_to_cpu = len(frame_names) > 1500
    # 存储中间状态
    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=async_loading_frames, offload_video_to_cpu=offload_video_to_cpu, offload_state_to_cpu=offload_state_to_cpu)
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None
    # 获取每个帧对象的mask
    input_palette, inputs_per_object = get_input_per_obj(base_video_dir=base_video_dir, input_mask_dir=input_mask_dir, video_name=video_name, use_all_masks=use_all_masks, per_obj_png_file=per_obj_png_file, track_object_appearing_later_in_video=track_object_appearing_later_in_video)
    # run inference separately for each object in the video
    # 进行GT Mask单目标对象拆分
    object_ids = sorted(inputs_per_object)
    if video_obj_list is not None:
        object_ids = [obj for obj in object_ids if obj in video_obj_list[video_name]]
    video_interaction_stats = create_empty_interaction_stats(video_name=video_name)
    obj_gt_mask_per_object = load_object_annotations(gt_mask_dir, frame_names, object_ids, per_obj_png_file=per_obj_png_file)
    finish_process_obj = set()
    # cur_pass_num = 1 if pass_num is not None else None
    # cur_pass_num：当前这一轮评估中，每个 object 最多允许发生多少次“用户纠错事件”
    if pass_num is not None and pass_num > 0:
        cur_pass_num = 2
    elif pass_num is not None and pass_num == 0:
        cur_pass_num = 0
    else:
        cur_pass_num = None
    while True:
        # 初始化要保存的结果
        # 记录object在哪些帧使用correction
        output_prompt_per_object = {obj_idx:{} for obj_idx in object_ids}
        # SAM自己估计的IoU，不一定准
        output_pred_iou_per_object = {obj_idx: [] for obj_idx in object_ids}
        # 给每个 object 保存IoU (和GT计算得来的)
        output_gt_iou_per_object = {obj_idx: [] for obj_idx in object_ids}
        video_segments = defaultdict(dict)
        for object_idx, object_id in enumerate(object_ids):
            if object_id in finish_process_obj:
                continue
            if LIT_LoRA_mode:
                # 输出： 每一帧的真实IOU，SAM预测的IOU，哪些帧进行了纠错
                output_gt_iou_per_object[object_id], output_pred_iou_per_object[object_id], output_prompt_per_object[object_id], finish_process = \
                    vos_online_one_pass_lora_correction(
                    predictor,
                    inference_state,
                    inputs_per_object,
                    obj_gt_mask_per_object,
                    video_segments,
                    object_id,
                    score_thresh,
                    correct_threshold,
                    correction_num_limit=cur_pass_num,
                    max_num_click_per_frame=max_num_click_per_frame,
                    training_epoch=training_epoch,
                )
            else:
                output_gt_iou_per_object[object_id], output_pred_iou_per_object[object_id], output_prompt_per_object[object_id], finish_process = \
                    vos_online_one_pass_user_correction(
                    predictor,
                    inference_state,
                    inputs_per_object,
                    obj_gt_mask_per_object,
                    video_segments,
                    object_id,
                    score_thresh,
                    correct_threshold,
                    correction_num_limit=cur_pass_num,
                    max_num_click_per_frame=max_num_click_per_frame,
                )
            if finish_process:
                finish_process_obj.add(object_id)
        accumulate_interaction_stats(video_interaction_stats, output_prompt_per_object)
        cache = {'pred_iou': output_pred_iou_per_object,
                'prompt': output_prompt_per_object, 'gt_iou': output_gt_iou_per_object}
        output_palette = input_palette or DAVIS_PALETTE
        if len(video_segments) == 0:
            break
        # write the output masks as palette PNG files to output_mask_dir
        if cur_pass_num != None:
            save_mask_dir = os.path.join(output_mask_dir, 'pred_mask', f'pass_{cur_pass_num}')
            if not os.path.exists(save_mask_dir):
                os.makedirs(save_mask_dir)
        else:
            save_mask_dir = os.path.join(output_mask_dir, 'pred_mask')
            if not os.path.exists(save_mask_dir):
                os.makedirs(save_mask_dir)
        for frame_idx, per_obj_output_mask in video_segments.items():
            save_masks_to_dir(
                output_mask_dir=save_mask_dir,
                video_name=video_name,
                frame_name=frame_names[frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=True,
                output_palette=output_palette,
            )
        if cur_pass_num != None and cur_pass_num > 0:
            cache_path = (os.path.join(output_mask_dir, 'cache', f'pass_{cur_pass_num}'))
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
        else:
            cache_path = (os.path.join(output_mask_dir, 'cache'))
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
        cache_path = os.path.join(cache_path, video_name + '.jsonl')
        with open(cache_path, 'w') as f:
            f.write(str(cache))
        predictor.reset_state(inference_state)
        if len(finish_process_obj) == len(object_ids) or cur_pass_num == None:
            break
        # with limited number of passes
        cur_pass_num += 2
        if cur_pass_num > pass_num:
            break
    if hasattr(inference_state["images"], "clear"):
        inference_state["images"].clear()
    predictor.reset_lora()
    del inference_state
    del output_prompt_per_object
    gc.collect()
    torch.cuda.empty_cache()
    return video_interaction_stats


def main():
    # Dataset: "lvosv2", "vost", "sav_val", "sav_test"
    DATASET = "sav_val"
    dataset_config = get_dataset_config(DATASET)

    # One-click VOS evaluation configuration. CLI arguments can still override these defaults.
    one_click_config = {
        "model_backend": "sam3",
        "sam3_checkpoint": "/opt/data/private/code/pretrain/sam3/sam3.pt",
        "device": "cuda",
        "sam3_temporal_disambiguation": True,
        "base_video_dir": dataset_config["base_video_dir"],
        "input_mask_dir": dataset_config["input_mask_dir"],
        "video_list_file": dataset_config["video_list_file"],
        "per_obj_png_file": dataset_config["per_obj_png_file"],
        "track_object_appearing_later_in_video": dataset_config["track_object_appearing_later_in_video"],
        "score_thresh": 0.0,
        "online_evaluation_unlimited": True,
        "num_pass": 1,
        "correct_threshold": 0.75,
        "LIT_LoRA_mode": True,
        "max_num_click_per_frame": 3,
        "training_epoch": 40,
        "offload_video_to_cpu": None,
        "offload_state_to_cpu": False,
        "async_loading_frames": False,
        "eval_jf": True,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_backend", choices=("sam3",), default=one_click_config["model_backend"], help="video model backend (default: sam3)")
    parser.add_argument("--sam3_checkpoint", type=str, default=one_click_config["sam3_checkpoint"], help="Meta SAM 3 checkpoint file or directory containing sam3.pt")
    parser.add_argument("--device", type=str, default=one_click_config["device"], help="model device; the upstream SAM 3 tracker currently requires CUDA")
    parser.add_argument("--sam3_temporal_disambiguation", action=argparse.BooleanOptionalAction, default=one_click_config["sam3_temporal_disambiguation"], help="enable SAM 3 temporal memory selection")
    parser.add_argument("--offload_video_to_cpu", action=argparse.BooleanOptionalAction, default=one_click_config["offload_video_to_cpu"], help="force CPU frame storage; default: GPU for <=1500 frames, CPU for longer videos")
    parser.add_argument("--offload_state_to_cpu", action=argparse.BooleanOptionalAction, default=one_click_config["offload_state_to_cpu"], help="keep tracker state on CPU (default: GPU, matching the original LIT code)")
    parser.add_argument("--async_loading_frames", action=argparse.BooleanOptionalAction, default=one_click_config["async_loading_frames"], help="preload all frames asynchronously (default: synchronous full-video loading)")
    parser.add_argument("--base_video_dir", type=str, default=one_click_config["base_video_dir"], help="directory containing videos to run VOS prediction on")
    parser.add_argument("--input_mask_dir", type=str, default=one_click_config["input_mask_dir"], help="directory containing input masks of each video")
    parser.add_argument("--video_list_file", type=str, default=one_click_config["video_list_file"], help="text file containing the list of video names to run VOS prediction on")
    parser.add_argument("--output_mask_dir", type=str, default=None, help="directory to save output masks; default is generated from dataset, method and threshold")
    parser.add_argument("--score_thresh", type=float, default=one_click_config["score_thresh"], help="threshold for the output mask logits (default: 0.0)")
    parser.add_argument("--use_all_masks", action="store_true", help="use all available GT masks as model inputs instead of only the first available mask per object")
    parser.add_argument("--per_obj_png_file", action=argparse.BooleanOptionalAction, default=one_click_config["per_obj_png_file"], help="whether GT masks are stored as separate per-object PNG files")
    parser.add_argument("--apply_postprocessing", action="store_true", help="whether to apply postprocessing such as hole filling")
    parser.add_argument("--track_object_appearing_later_in_video", action=argparse.BooleanOptionalAction, default=one_click_config["track_object_appearing_later_in_video"], help="whether to track objects whose first annotation appears later in the video")
    parser.add_argument("--use_vos_optimized_video_predictor", action="store_true", help="whether to use VOS optimized video predictor with all modules compiled")
    parser.add_argument("--online_evaluation_unlimited", action=argparse.BooleanOptionalAction, default=one_click_config["online_evaluation_unlimited"], help="whether to use online evaluation with unlimited correction")
    parser.add_argument("--num_pass", type=int, default=one_click_config["num_pass"], help="number of passes for online evaluation")
    parser.add_argument("--correct_threshold", type=float, default=one_click_config["correct_threshold"], help="IoU threshold that triggers correction")
    parser.add_argument("--LIT_LoRA_mode", action=argparse.BooleanOptionalAction, default=one_click_config["LIT_LoRA_mode"], help="whether to use LIT-LoRA")
    parser.add_argument("--max_num_click_per_frame", type=int, default=one_click_config["max_num_click_per_frame"], help="maximum number of point clicks per corrected frame")
    parser.add_argument("--training_epoch", type=int, default=one_click_config["training_epoch"], help="number of training epochs for LoRA")
    parser.add_argument("--video_obj_list_file", type=str, default=None, help="file containing the list of video and object pairs to run VOS prediction on")
    parser.add_argument("--eval_jf", action=argparse.BooleanOptionalAction, default=one_click_config["eval_jf"])
    args = parser.parse_args()

    args.dataset = DATASET
    if args.output_mask_dir is None:
        method_name = "LoRA" if args.LIT_LoRA_mode else "NoLoRA"
        threshold_name = f"{args.correct_threshold:g}".replace(".", "")
        args.output_mask_dir = f"/opt/data/private/out/{DATASET}/{method_name}_t{threshold_name}"

    print(f"Dataset: {args.dataset}")
    print(f"Video directory: {args.base_video_dir}")
    print(f"GT mask directory: {args.input_mask_dir}")
    print(f"Video list: {args.video_list_file}")
    print(f"Per-object GT PNG: {args.per_obj_png_file}")
    print(f"Track later-appearing objects: {args.track_object_appearing_later_in_video}")
    print(f"Output directory: {args.output_mask_dir}")

    if args.use_vos_optimized_video_predictor:
        parser.error("--use_vos_optimized_video_predictor is not supported with the SAM 3 backend.")

    from lit_sam3 import build_lit_sam3_video_predictor

    predictor = build_lit_sam3_video_predictor(checkpoint_path=args.sam3_checkpoint, device=args.device, apply_temporal_disambiguation=args.sam3_temporal_disambiguation, apply_postprocessing=args.apply_postprocessing, non_overlap_masks_for_output=not args.per_obj_png_file)

    if args.LIT_LoRA_mode:
        predictor.LIT_LoRA_mode = True
        predictor.reset_lora()
    else:
        predictor.LIT_LoRA_mode = False

    if args.online_evaluation_unlimited:
        args.num_pass = None

    if args.use_all_masks:
        print(f"using all available masks in input_mask_dir as input to the {args.model_backend.upper()} model")
    else:
        print(f"using only the first available mask per object in input_mask_dir as input to the {args.model_backend.upper()} model")

    if args.video_list_file is not None:
        with open(args.video_list_file, "r", encoding="utf-8") as f:
            video_names = [v.strip() for v in f if v.strip()]
    else:
        video_names = [p for p in os.listdir(args.base_video_dir) if os.path.isdir(os.path.join(args.base_video_dir, p))]

    video_names = sorted(video_names)
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    os.makedirs(args.output_mask_dir, exist_ok=True)
    with open(os.path.join(args.output_mask_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2, ensure_ascii=False)

    if args.video_obj_list_file is not None:
        with open(args.video_obj_list_file, "r", encoding="utf-8") as f:
            video_obj_list = json.load(f)
        video_names = sorted(video_obj_list.keys())
    else:
        video_obj_list = None

    memory_results = []
    interaction_results = []

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on '{video_name}' video.")
        with report_video_peak_memory(video_name, args.device) as memory_stats:
            interaction_stats = vos_online_evaluation_pass(predictor=predictor, base_video_dir=args.base_video_dir, input_mask_dir=args.input_mask_dir, output_mask_dir=args.output_mask_dir, video_name=video_name, score_thresh=args.score_thresh, use_all_masks=args.use_all_masks, per_obj_png_file=args.per_obj_png_file, LIT_LoRA_mode=args.LIT_LoRA_mode, correct_threshold=args.correct_threshold, track_object_appearing_later_in_video=args.track_object_appearing_later_in_video, pass_num=args.num_pass, max_num_click_per_frame=args.max_num_click_per_frame, video_obj_list=video_obj_list, training_epoch=args.training_epoch, offload_video_to_cpu=args.offload_video_to_cpu, offload_state_to_cpu=args.offload_state_to_cpu, async_loading_frames=args.async_loading_frames)
        memory_results.append(memory_stats.copy())
        interaction_results.append(interaction_stats)

    print(f"completed VOS prediction on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")

    total_corrections = sum(item["user_corrections"] for item in interaction_results)
    total_clicks = sum(item["user_clicks"] for item in interaction_results)
    total_point_corrections = sum(item["point_corrections"] for item in interaction_results)
    total_mask_corrections = sum(item["mask_corrections"] for item in interaction_results)
    total_lora_auto_corrections = sum(item["lora_auto_corrections"] for item in interaction_results)
    num_completed_videos = len(interaction_results)

    if num_completed_videos > 0:
        avg_corrections_per_video = total_corrections / num_completed_videos
        avg_clicks_per_video = total_clicks / num_completed_videos
    else:
        avg_corrections_per_video = 0.0
        avg_clicks_per_video = 0.0

    if total_corrections > 0:
        avg_clicks_per_correction = total_clicks / total_corrections
        mask_correction_ratio = total_mask_corrections / total_corrections
    else:
        avg_clicks_per_correction = 0.0
        mask_correction_ratio = 0.0

    print("\n" + "=" * 70)
    print("FINAL INTERACTION SUMMARY")
    print("=" * 70)
    print(f"{'Videos completed':<32}: {num_completed_videos} / {len(video_names)}")
    print("-" * 70)
    print(f"{'Total user corrections':<32}: {total_corrections}")
    print(f"{'Avg. corrections / video':<32}: {avg_corrections_per_video:.3f}")
    print("-" * 70)
    print(f"{'Total user clicks':<32}: {total_clicks}")
    print(f"{'Avg. clicks / video':<32}: {avg_clicks_per_video:.3f}")
    print(f"{'Avg. clicks / correction':<32}: {avg_clicks_per_correction:.3f}")
    print("-" * 70)
    print(f"{'Point corrections':<32}: {total_point_corrections}")
    print(f"{'Mask corrections':<32}: {total_mask_corrections}")
    print(f"{'Mask correction ratio':<32}: {mask_correction_ratio * 100:.2f}%")
    if args.LIT_LoRA_mode:
        print(f"{'LoRA auto corrections':<32}: {total_lora_auto_corrections}")
    print("=" * 70)

    print("FINAL RESOURCE SUMMARY")
    print("=" * 70)
    print(f"{'Videos completed':<30}: {len(memory_results)} / {len(video_names)}")
    for key, label, per_video in (("cpu_rss_peak_gib", "CPU RSS peak", "/video"), ("gpu_allocated_peak_gib", "GPU allocated peak", ""), ("gpu_reserved_peak_gib", "GPU reserved peak", "")):
        print("-" * 70)
        results = [result for result in memory_results if result[key] is not None]
        maximum_label = f"{label} (maximum)"
        mean_label = f"{label} (mean{per_video})"
        median_label = f"{label} (median{per_video})"
        if results:
            maximum = max(results, key=lambda result: result[key])
            values = [result[key] for result in results]
            print(f"{maximum_label:<30}: {maximum[key]:.3f} GiB ({maximum['video_name']})")
            print(f"{mean_label:<30}: {np.mean(values):.3f} GiB")
            print(f"{median_label:<30}: {np.median(values):.3f} GiB")
        else:
            print(f"{maximum_label:<30}: N/A")
            print(f"{mean_label:<30}: N/A")
            print(f"{median_label:<30}: N/A")
    print("=" * 70, flush=True)

    interaction_df = pd.DataFrame(interaction_results)
    interaction_csv_path = os.path.join(args.output_mask_dir, "interaction_per_video.csv")
    interaction_df.to_csv(interaction_csv_path, index=False)

    interaction_summary = {
        "dataset": args.dataset,
        "num_videos": num_completed_videos,
        "total_corrections": total_corrections,
        "avg_corrections_per_video": avg_corrections_per_video,
        "total_clicks": total_clicks,
        "avg_clicks_per_video": avg_clicks_per_video,
        "avg_clicks_per_correction": avg_clicks_per_correction,
        "point_corrections": total_point_corrections,
        "mask_corrections": total_mask_corrections,
        "mask_correction_ratio": mask_correction_ratio,
        "lora_auto_corrections": total_lora_auto_corrections,
        "correct_threshold": args.correct_threshold,
        "max_num_click_per_frame": args.max_num_click_per_frame,
        "LIT_LoRA_mode": args.LIT_LoRA_mode,
        "model_backend": args.model_backend,
    }

    interaction_json_path = os.path.join(args.output_mask_dir, "interaction_summary.json")
    with open(interaction_json_path, "w", encoding="utf-8") as f:
        json.dump(interaction_summary, f, indent=2, ensure_ascii=False)

    print(f"\nInteraction per-video CSV saved to: {interaction_csv_path}")
    print(f"Interaction summary JSON saved to: {interaction_json_path}")

    if args.eval_jf:
        pred_dir = os.path.join(args.output_mask_dir, "pred_mask")
        evaluate_jf(input_mask_dir=args.input_mask_dir, pred_dir=pred_dir, output_dir=args.output_mask_dir)


if __name__ == "__main__":
    main()
