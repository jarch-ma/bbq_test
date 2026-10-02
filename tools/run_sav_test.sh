#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/opt/data/private/code/pretrain/sam3/sam3.pt}"
EVAL_SCRIPT="$PROJECT_DIR/tools/online_eval.py"
DATA_DIR="/home/jiaqi/Data/LIT_Data/SA-V/sav_test"
VIDEO_LIST="$DATA_DIR/sav_test.txt"
BASELINE_OUTPUT="$PROJECT_DIR/results/SAV_test_baseline"
LORA_OUTPUT="$PROJECT_DIR/results/SAV_test_LIT"
BASELINE_LOG="$PROJECT_DIR/tools/SAV_test_baseline.log"
LORA_LOG="$PROJECT_DIR/tools/SAV_test_Lora.log"
STATUS_LOG="$PROJECT_DIR/tools/SAV_test_runner.log"
COMPLETION_TEXT="completed VOS prediction on 150 videos"

log_event() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" >> "$STATUS_LOG"
}

run_experiment() {
    local mode="$1"
    local output_dir="$2"
    local log_file="$3"
    local lora_flag="$4"

    if grep -q "$COMPLETION_TEXT" "$log_file" 2>/dev/null; then
        log_event "$mode is already complete; skipping."
        return 0
    fi

    if [[ -e "$output_dir" || -e "$log_file" ]]; then
        log_event "$mode has incomplete existing output; refusing to overwrite $output_dir or $log_file."
        return 2
    fi

    log_event "Starting $mode."
    env MPLCONFIGDIR=/tmp/lit-matplotlib "$PYTHON_BIN" -u "$EVAL_SCRIPT" \
        --model_backend sam3 \
        --sam3_checkpoint "$SAM3_CHECKPOINT" \
        --base_video_dir "$DATA_DIR/JPEGImages_24fps" \
        --input_mask_dir "$DATA_DIR/Annotations_6fps" \
        --video_list_file "$VIDEO_LIST" \
        --output_mask_dir "$output_dir" \
        --per_obj_png_file \
        --no-track_object_appearing_later_in_video \
        --online_evaluation_unlimited \
        --correct_threshold 0.5 \
        "$lora_flag" \
        > "$log_file" 2>&1
    local status=$?
    log_event "$mode exited with status $status."
    return "$status"
}

log_event "SA-V Test runner started."

run_experiment "SA-V Test baseline" "$BASELINE_OUTPUT" "$BASELINE_LOG" "--no-LIT_LoRA_mode" || exit $?
run_experiment "SA-V Test LIT-LoRA" "$LORA_OUTPUT" "$LORA_LOG" "--LIT_LoRA_mode" || exit $?

log_event "All SA-V Test experiments completed."
