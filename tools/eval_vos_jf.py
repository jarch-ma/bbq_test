import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ================================================================
# Official evaluator paths
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LVOS_EVAL_DIR = PROJECT_ROOT / "configs" / "lvos_evaluation"
VOST_EVAL_DIR = PROJECT_ROOT / "configs" / "vost_evaluation"
SAV_EVAL_DIR = PROJECT_ROOT / "configs" / "sav_evaluation"


LVOS_SET = "valid"
VOST_SET = "val"
MP_NUMS = 4


# ================================================================
# Common utilities
# ================================================================

def detect_dataset(input_mask_dir):
    path = str(input_mask_dir).lower().replace("-", "").replace("_", "")

    if "lvos" in path:
        return "lvos"
    if "vost" in path:
        return "vost"
    if "sav" in path:
        return "sav"

    raise ValueError(f"Cannot automatically detect dataset from path: {input_mask_dir}\nSupported datasets: LVOS, SA-V, VOST.")

def frame_sort_key(name):
    stem = Path(name).stem

    try:
        return int(stem.replace("frame", "").split("_")[-1])
    except ValueError:
        return stem


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_csv_clean(path):
    table = pd.read_csv(path)
    table.columns = [str(column).strip() for column in table.columns]
    return table


def run_command(command, cwd):
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL EVALUATOR")
    print("=" * 70)
    print(" ".join(str(x) for x in command))
    print("=" * 70)

    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Official evaluation failed with exit code {error.returncode}.") from error


# ================================================================
# LVOS
# ================================================================

def merge_per_object_predictions(pred_dir, merged_dir):
    """
    Convert:

        pred_mask/
            video/
                001/
                    frame.png
                002/
                    frame.png

    into:

        merged_results/
            video/
                frame.png

    Pixel values:
        0 = background
        1,2,3,... = object IDs
    """

    pred_dir = Path(pred_dir)
    merged_dir = Path(merged_dir)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    merged_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = sorted(path for path in pred_dir.iterdir() if path.is_dir())

    if not sequence_dirs:
        raise RuntimeError(f"No video prediction directories found in: {pred_dir}")

    total_sequences = 0
    total_frames = 0

    print("\n" + "=" * 70)
    print("PREPARING PACKED MASKS")
    print("=" * 70)

    for sequence_dir in sequence_dirs:
        object_dirs = []

        for object_dir in sequence_dir.iterdir():
            if not object_dir.is_dir():
                continue

            try:
                object_id = int(object_dir.name)
            except ValueError:
                continue

            if object_id < 0:
                continue

            if object_id >= 255:
                raise ValueError(f"Invalid object ID {object_id} in {sequence_dir.name}; 255 is reserved.")

            object_dirs.append((object_id, object_dir))

        if not object_dirs:
            print(f"[warning] No object directories: {sequence_dir}")
            continue

        object_dirs.sort(key=lambda x: x[0])

        frame_names = set()

        for _, object_dir in object_dirs:
            frame_names.update(path.name for path in object_dir.glob("*.png"))

        frame_names = sorted(frame_names, key=frame_sort_key)

        if not frame_names:
            continue

        output_sequence_dir = merged_dir / sequence_dir.name
        output_sequence_dir.mkdir(parents=True, exist_ok=True)

        for frame_name in frame_names:
            frame_shape = None
            object_masks = []

            for object_id, object_dir in object_dirs:
                mask_path = object_dir / frame_name

                if not mask_path.exists():
                    continue

                mask = np.array(Image.open(mask_path))

                if mask.ndim == 3:
                    mask = mask[..., 0]

                binary_mask = mask > 0

                if frame_shape is None:
                    frame_shape = binary_mask.shape
                elif binary_mask.shape != frame_shape:
                    raise ValueError(f"Mask shape mismatch: {sequence_dir.name}/{frame_name}")

                object_masks.append((object_id, binary_mask))

            if frame_shape is None:
                continue

            merged_mask = np.zeros(frame_shape, dtype=np.uint8)

            # 与 online_eval.py 的 overlap policy 保持一致：
            # 大 ID 先写，小 ID 后覆盖
            for object_id, binary_mask in sorted(object_masks, key=lambda x: x[0], reverse=True):
                merged_mask[binary_mask] = object_id

            Image.fromarray(merged_mask).save(output_sequence_dir / frame_name)
            total_frames += 1

        total_sequences += 1
        print(f"[merge] {sequence_dir.name}: {len(object_dirs)} objects, {len(frame_names)} frames")

    print("-" * 70)
    print(f"Sequences prepared : {total_sequences}")
    print(f"Frames prepared    : {total_frames}")
    print(f"Output             : {merged_dir}")
    print("=" * 70)

    return merged_dir


def evaluate_lvos(input_mask_dir, pred_dir, output_dir):
    input_mask_dir = Path(input_mask_dir)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)

    dataset_root = input_mask_dir.parent
    evaluator = LVOS_EVAL_DIR / "evaluation_method.py"

    if not evaluator.is_file():
        raise FileNotFoundError(f"LVOS evaluator not found: {evaluator}")

    work_dir = output_dir / "jf_eval"
    merged_dir = work_dir / "merged_results"

    merge_per_object_predictions(pred_dir, merged_dir)

    global_csv = merged_dir / f"global_results-{LVOS_SET}.csv"
    sequence_csv = merged_dir / f"per-sequence_results-{LVOS_SET}.csv"

    if global_csv.exists():
        global_csv.unlink()

    if sequence_csv.exists():
        sequence_csv.unlink()

    command = [
        sys.executable,
        str(evaluator),
        "--task", "semi-supervised",
        "--lvos_path", str(dataset_root),
        "--set", LVOS_SET,
        "--results_path", str(merged_dir),
        "--mp_nums", str(MP_NUMS),
        "--m_class", "mp",
    ]

    run_command(command, LVOS_EVAL_DIR)

    if not global_csv.exists():
        raise RuntimeError(f"LVOS evaluator did not generate: {global_csv}")

    table = read_csv_clean(global_csv)
    row = table.iloc[0]

    summary = {
        "dataset": "LVOSv2",
        "split": LVOS_SET,
        "J": float(row["J-Mean"]),
        "F": float(row["F-Mean"]),
        "J&F": float(row["J&F-Mean"]),
        "J_percent": float(row["J-Mean"]) * 100.0,
        "F_percent": float(row["F-Mean"]) * 100.0,
        "J&F_percent": float(row["J&F-Mean"]) * 100.0,
    }

    save_json(summary, output_dir / "vos_eval_summary.json")
    save_json(summary, output_dir / "jf_summary.json")

    shutil.copy2(global_csv, output_dir / "jf_global_results.csv")
    shutil.copy2(sequence_csv, output_dir / "jf_per_sequence_results.csv")

    print("\n" + "=" * 70)
    print("LVOS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"J   : {summary['J_percent']:.2f}")
    print(f"F   : {summary['F_percent']:.2f}")
    print(f"J&F : {summary['J&F_percent']:.2f}")
    print("=" * 70)

    return summary


# ================================================================
# SA-V
# ================================================================

def evaluate_sav(input_mask_dir, pred_dir, output_dir):
    input_mask_dir = Path(input_mask_dir)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)

    evaluator = SAV_EVAL_DIR / "sav_evaluator.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"SA-V evaluator not found: {evaluator}")

    command = [
        sys.executable,
        str(evaluator),
        "--gt_root", str(input_mask_dir),
        "--pred_root", str(pred_dir),
        "--num_processes", str(MP_NUMS),
        "--strict",
    ]

    run_command(command, evaluator.parent)

    results_csv = pred_dir / "results.csv"

    if not results_csv.exists():
        raise RuntimeError(f"SA-V evaluator did not generate: {results_csv}")

    table = read_csv_clean(results_csv)
    global_rows = table[table["sequence"].astype(str).str.strip() == "Global score"]

    if global_rows.empty:
        raise RuntimeError("Cannot find 'Global score' in SA-V results.csv")

    row = global_rows.iloc[0]

    j = float(row["J"])
    f = float(row["F"])
    jf = float(row["J&F"])

    summary = {
        "dataset": "SA-V",
        "protocol": "official_sav",
        "J": j / 100.0,
        "F": f / 100.0,
        "J&F": jf / 100.0,
        "J_percent": j,
        "F_percent": f,
        "J&F_percent": jf,
    }

    save_json(summary, output_dir / "vos_eval_summary.json")
    save_json(summary, output_dir / "jf_summary.json")

    shutil.copy2(results_csv, output_dir / "sav_results.csv")

    print("\n" + "=" * 70)
    print("SA-V EVALUATION SUMMARY")
    print("=" * 70)
    print(f"J   : {j:.2f}")
    print(f"F   : {f:.2f}")
    print(f"J&F : {jf:.2f}")
    print("=" * 70)

    return summary


# ================================================================
# VOST
# ================================================================
def evaluate_vost(input_mask_dir, pred_dir, output_dir):
    input_mask_dir = Path(input_mask_dir)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)

    dataset_root = input_mask_dir.parent
    evaluator = VOST_EVAL_DIR / "evaluation_method.py"

    if not evaluator.is_file():
        raise FileNotFoundError(f"VOST official evaluator not found: {evaluator}")

    work_dir = output_dir / "vost_eval"
    results_dir = work_dir / "results"

    merge_per_object_predictions(pred_dir, results_dir)

    global_csv = results_dir / f"global_results-{VOST_SET}.csv"
    sequence_csv = results_dir / f"per-sequence_results-{VOST_SET}.csv"

    command = [
        sys.executable,
        str(evaluator),
        "--set", VOST_SET,
        "--dataset_path", str(dataset_root),
        "--results_path", str(results_dir),
        "--re",
    ]

    run_command(command, VOST_EVAL_DIR)

    if not global_csv.exists():
        raise RuntimeError(f"VOST evaluator did not generate: {global_csv}")

    table = read_csv_clean(global_csv)
    row = table.iloc[0]

    summary = {
        "dataset": "VOST",
        "split": VOST_SET,
        "J-Mean": float(row["J-Mean"]),
        "J-Recall": float(row["J-Recall"]),
        "J-Decay": float(row["J-Decay"]),
        "J_last-Mean": float(row["J_last-Mean"]),
        "J_last-Recall": float(row["J_last-Recall"]),
        "J_last-Decay": float(row["J_last-Decay"]),
    }

    save_json(summary, output_dir / "vos_eval_summary.json")

    shutil.copy2(global_csv, output_dir / "vost_global_results.csv")
    shutil.copy2(sequence_csv, output_dir / "vost_per_sequence_results.csv")

    print("\n" + "=" * 70)
    print("VOST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"J-Mean      : {summary['J-Mean']:.4f}")
    print(f"J-Recall    : {summary['J-Recall']:.4f}")
    print(f"J-Decay     : {summary['J-Decay']:.4f}")
    print(f"J_last-Mean : {summary['J_last-Mean']:.4f}")
    print("=" * 70)

    return summary


# ================================================================
# Unified public interface
# ================================================================

def evaluate_jf(input_mask_dir, pred_dir, output_dir):
    dataset = detect_dataset(input_mask_dir)

    print("\n" + "=" * 70)
    print(f"Detected dataset: {dataset.upper()}")
    print("=" * 70)

    if dataset == "lvos":
        return evaluate_lvos(input_mask_dir, pred_dir, output_dir)

    if dataset == "sav":
        return evaluate_sav(input_mask_dir, pred_dir, output_dir)

    if dataset == "vost":
        return evaluate_vost(input_mask_dir, pred_dir, output_dir)

    raise ValueError(f"Unsupported dataset: {dataset}")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mask_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    evaluate_jf(input_mask_dir=args.input_mask_dir, pred_dir=args.pred_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()