from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

sam3_path = "/home/jiaqi/Code/Pretrain/sam3"
video_path = "/home/jiaqi/Data/LIT_Data/bedroom.mp4"
save_path = "/home/jiaqi/Data/LIT_Data/bedroom_sam3_mask.mp4"

device = Accelerator().device
print("Device:", device)

model = Sam3VideoModel.from_pretrained(sam3_path).to(device, dtype=torch.bfloat16)
model.eval()
processor = Sam3VideoProcessor.from_pretrained(sam3_path)

video_frames, video_metadata = load_video(video_path)
print("Video shape:", video_frames.shape)
print("Total frames:", len(video_frames))

inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device=device,
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

text = "girl"
inference_session = processor.add_text_prompt(inference_session=inference_session, text=text)
print("Prompt:", text)

def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x

def prepare_frame(frame):
    frame = to_numpy(frame)
    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4] and frame.shape[-1] not in [1, 3, 4]:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame

def get_color(object_id):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0)
    ]
    return colors[int(object_id) % len(colors)]

def visualize_prediction(frame_idx, processed_outputs):
    frame = prepare_frame(video_frames[frame_idx])
    masks = to_numpy(processed_outputs["masks"])
    boxes = to_numpy(processed_outputs["boxes"])
    scores = to_numpy(processed_outputs["scores"])
    object_ids = to_numpy(processed_outputs["object_ids"])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame)

    for i in range(len(object_ids)):
        object_id = int(object_ids[i])
        color = np.array(get_color(object_id)) / 255.0

        mask = masks[i].squeeze().astype(bool)
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[..., :3] = color
        overlay[..., 3] = mask.astype(np.float32) * 0.45
        ax.imshow(overlay)

        x1, y1, x2, y2 = boxes[i]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"))
        ax.text(x1, max(float(y1) - 5, 5), f"ID={object_id} score={float(scores[i]):.3f}",
                fontsize=11, color="white", bbox=dict(facecolor=color, alpha=0.8, edgecolor="none"))

    ax.set_title(f"Frame {frame_idx}    Objects={len(object_ids)}")
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

outputs_per_frame = {}

for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session,
    max_frame_num_to_track=50
):
    frame_idx = model_outputs.frame_idx
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[frame_idx] = processed_outputs

    print("=" * 60)
    print(f"Frame: {frame_idx}")
    print(f"Object IDs: {processed_outputs['object_ids'].tolist()}")
    print(f"Scores: {processed_outputs['scores'].tolist()}")
    print(f"Boxes shape: {processed_outputs['boxes'].shape}")
    print(f"Masks shape: {processed_outputs['masks'].shape}")

    visualize_prediction(frame_idx, processed_outputs)

print(f"Processed {len(outputs_per_frame)} frames")

first_frame = prepare_frame(video_frames[0])
height, width = first_frame.shape[:2]
fps = 30

if hasattr(video_metadata, "fps"):
    fps = float(video_metadata.fps)

writer = cv2.VideoWriter(
    save_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

for frame_idx in sorted(outputs_per_frame.keys()):
    frame = prepare_frame(video_frames[frame_idx])
    outputs = outputs_per_frame[frame_idx]

    masks = to_numpy(outputs["masks"])
    boxes = to_numpy(outputs["boxes"])
    scores = to_numpy(outputs["scores"])
    object_ids = to_numpy(outputs["object_ids"])

    result = frame.copy()

    for i in range(len(object_ids)):
        object_id = int(object_ids[i])
        color = np.array(get_color(object_id), dtype=np.uint8)
        mask = masks[i].squeeze().astype(bool)
        result[mask] = (result[mask] * 0.55 + color * 0.45).astype(np.uint8)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    for i in range(len(object_ids)):
        object_id = int(object_ids[i])
        score = float(scores[i])
        color_rgb = get_color(object_id)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(result, f"ID={object_id} {score:.3f}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

    cv2.putText(result, f"Frame {frame_idx}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    writer.write(result)

writer.release()
print("Finished")
print(f"Saved video: {save_path}")