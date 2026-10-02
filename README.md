<div align="center">
<h1>Live Interactive Training for Video Segmentation</h1>

<h2>CVPR 2026</h2>

<a href="https://arxiv.org/abs/2603.26929"><img src="https://img.shields.io/badge/arXiv-2603.26929-b31b1b" alt="arXiv"></a>
<a href="https://youngxinyu1802.github.io/projects/LIT/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>



[Xinyu Yang](https://youngxinyu1802.github.io/), [Haozheng Yu](https://haozheng-yu.github.io/), [Yihong Sun](https://yihongsun.github.io/), [Bharath Hariharan](https://www.cs.cornell.edu/~bharathh/), [Jennifer J. Sun](https://jenjsun.com/)

Cornell University
</div>

<img width="903" height="530" alt="image" src="https://github.com/user-attachments/assets/ece71465-cb97-42bc-aaa3-9e5ec0bf4752" />


## Overview

**LIT (Live Interactive Training)** is an online adaptation method for interactive video object segmentation built on top of [SAM 3](https://github.com/facebookresearch/sam3). Standard SAM 3 correction workflows apply user clicks frame-by-frame without updating the model, so mistakes tend to recur across the rest of the video. LIT instead treats each user correction as a training signal: as clicks come in, LIT fine-tunes lightweight LoRA adapters on the fly, so the model *keeps improving on the current video as it is being corrected*, reducing the number of corrections needed on later frames.

This repo provides:
- **LIT-LoRA mode**: online inference with on-the-fly LoRA adaptation from user correction clicks.
- **Baseline mode**: standard SAM 3 online inference with correction clicks but no adaptation, for comparison.
- **SAM 3 backend**: the sole model backend, with an adapter that runs the same mask-prompt VOS, correction, and LIT-LoRA workflow on the SAM 3 tracker.
- An `online_eval.py` tool that simulates a user by sampling correction clicks from ground-truth masks.


## Installation

### Environment
```
conda create -n LIT python=3.10
conda activate LIT
conda install -c conda-forge ffmpeg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ./sam3 -e ".[notebooks]"
```
### Checkpoint

Place the Meta-format `sam3.pt` at `/opt/data/private/code/pretrain/sam3/sam3.pt`
(the current default), or pass its path with `--sam3_checkpoint`.
Install both local packages using the command above: `sam3` supplies the model,
while `lit-sam3` supplies the existing LIT correction and LoRA training workflow.
No separate legacy model package or CUDA extension is required.

## Example Usage

`tools/online_eval.py` runs online video segmentation with simulated user corrections: at each step, correction clicks are sampled from the ground-truth mask (used as an oracle for the user) wherever the predicted mask disagrees with it beyond `--correct_threshold`. Prepare frames under `<dataset>/JPEGImages/<video>/` and annotations under `<dataset>/Annotations/<video>/`. The example commands below use `LIT_dataset/example/` as a placeholder; populate it or substitute your dataset path.

### SAM 3 backend

Install the official SAM 3 package in the same environment, then pass Meta's
`sam3.pt` checkpoint (a directory containing that file is also accepted):

```bash
python ./tools/online_eval.py \
  --model_backend sam3 \
  --sam3_checkpoint /opt/data/private/code/pretrain/sam3/sam3.pt \
  --base_video_dir ./LIT_dataset/example/JPEGImages \
  --input_mask_dir ./LIT_dataset/example/Annotations \
  --output_mask_dir ./results/sam3_lit \
  --online_evaluation_unlimited \
  --correct_threshold 0.5 \
  --LIT_LoRA_mode
```

The adapter loads only the SAM 3 tracker and visual backbone because the text
encoder and open-vocabulary detector are not used in this mask-prompt VOS path.
The upstream SAM 3 tracker currently requires CUDA. Use the Meta-format
`sam3.pt`; this loader does not accept the Hugging Face `model.safetensors`
layout.

### LIT-LoRA mode
```bash
python ./tools/online_eval.py \
  --model_backend sam3 \
  --sam3_checkpoint /opt/data/private/code/pretrain/sam3/sam3.pt \
  --base_video_dir ./LIT_dataset/example/JPEGImages \
  --input_mask_dir ./LIT_dataset/example/Annotations \
  --output_mask_dir ./results \
  --online_evaluation_unlimited \
  --correct_threshold 0.5 \
  --LIT_LoRA_mode
```

### Baseline (without LIT-LoRA mode)
```bash
python ./tools/online_eval.py \
  --model_backend sam3 \
  --sam3_checkpoint /opt/data/private/code/pretrain/sam3/sam3.pt \
  --base_video_dir ./LIT_dataset/example/JPEGImages \
  --input_mask_dir ./LIT_dataset/example/Annotations \
  --output_mask_dir ./results \
  --online_evaluation_unlimited \
  --correct_threshold 0.5 \
  --no-LIT_LoRA_mode
```

### Key arguments
| Argument | Description |
| --- | --- |
| `--model_backend` | Only `sam3` is supported (the default). |
| `--sam3_checkpoint` | Meta SAM 3 checkpoint file, or a directory containing `sam3.pt`. |
| `--sam3_temporal_disambiguation` | Enable SAM 3 temporal memory selection (enabled by default). |
| `--offload_video_to_cpu` | By default, keep full videos on GPU up to 1500 frames, and on CPU above 1500; this flag explicitly forces CPU storage. |
| `--offload_state_to_cpu` | Move tracking state to CPU; disabled by default, so states remain on GPU. |
| `--async_loading_frames` | Preload the entire video asynchronously; disabled by default (synchronous loading). |
| `--LIT_LoRA_mode` | Enable on-the-fly LIT-LoRA adaptation from correction clicks (use `--no-LIT_LoRA_mode` for the baseline). |
| `--correct_threshold` | IoU threshold below which a frame is corrected (default: `0.5`). |
| `--max_num_click_per_frame` | Max simulated correction clicks per frame (default: `3`). |
| `--training_epoch` | Number of LoRA fine-tuning epochs per correction (default: `40`). |
| `--online_evaluation_unlimited` | Allow unlimited correction passes over the video instead of a fixed `--num_pass`. |

`tools/vos_inference.py` also uses SAM 3 for plain VOS inference, without online correction. It accepts `--sam3_checkpoint`; YAML model configurations are no longer needed.

Frame folders are loaded and retained in full as normalized float32 tensors.
The default evaluation policy matches the original LIT code: videos with at most
1500 frames stay on GPU, longer videos stay on CPU, and tracking state stays on
GPU. Explicit `--offload_video_to_cpu` / `--no-offload_video_to_cpu` flags override
the automatic frame-storage policy. All GT annotations are read once and their
per-object masks are retained in CPU memory. There is no 32-frame or two-GT cache
limit. The SAM 3 normalization, correction and LoRA training logic are unchanged.

Run `python ./tools/online_eval.py --help` for the full list of options.

---

This repo is built on top of [SAM 3](https://github.com/facebookresearch/sam3) by Meta AI. We thank the SAM 3 team for making their code and models publicly available.

# Citing
```bibtex
@inproceedings{yang2026live,
  title={Live Interactive Training for Video Segmentation},
  author={Yang, Xinyu and Yu, Haozheng and Sun, Yihong and Hariharan, Bharath and Sun, Jennifer J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={39827--39837},
  year={2026}
}
```
