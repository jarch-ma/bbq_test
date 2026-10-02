#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="/home/jiaqi/Code/LIT-LoRA"
PYTHON_BIN="/home/jiaqi/miniconda3/envs/LIT/bin/python"
EVAL_SCRIPT="$PROJECT_DIR/tools/online_eval.py"
DATA_DIR="/home/jiaqi/Data/LIT_Data/SA-V/sav_val"
BASELINE_LOG="$PROJECT_DIR/tools/SAV_val_baseline.log"
LORA_LOG="$PROJECT_DIR/tools/SAV_val_Lora.log"
SCHEDULER_LOG="$PROJECT_DIR/tools/SAV_val_scheduler.log"
BASELINE_OUTPUT="$PROJECT_DIR/results/SAV_val_baseline"
LORA_OUTPUT="$PROJECT_DIR/results/SAV_val_LIT"
VIDEO_LIST="$DATA_DIR/sav_val.txt"
TARGET_TIME="$(date -d 'tomorrow 03:00' '+%Y-%m-%d %H:%M:%S')"

log_event() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" >> "$SCHEDULER_LOG"
}

run_baseline() {
    log_event "Baseline is incomplete and no baseline process is alive; restarting it."
    env MPLCONFIGDIR=/tmp/lit-matplotlib "$PYTHON_BIN" -u "$EVAL_SCRIPT" \
        --base_video_dir "$DATA_DIR/JPEGImages_24fps" \
        --input_mask_dir "$DATA_DIR/Annotations_6fps" \
        --video_list_file "$VIDEO_LIST" \
        --output_mask_dir "$BASELINE_OUTPUT" \
        --per_obj_png_file \
        --no-track_object_appearing_later_in_video \
        --online_evaluation_unlimited \
        --correct_threshold 0.5 \
        --no-LIT_LoRA_mode \
        >> "$BASELINE_LOG" 2>&1
}

log_event "Scheduler started; LoRA activation target is $TARGET_TIME."
sleep_seconds=$(( $(date -d "$TARGET_TIME" '+%s') - $(date '+%s') ))
if (( sleep_seconds > 0 )); then
    sleep "$sleep_seconds"
fi
log_event "03:00 activation reached."

while ! grep -q "completed VOS prediction on 155 videos" "$BASELINE_LOG" 2>/dev/null; do
    if pgrep -f "$EVAL_SCRIPT.*SAV_val_baseline" >/dev/null; then
        log_event "Baseline is still running; checking again in 5 minutes."
        sleep 300
    else
        run_baseline
        baseline_status=$?
        if (( baseline_status != 0 )); then
            log_event "Baseline restart exited with status $baseline_status; retrying in 10 minutes."
            sleep 600
        fi
    fi
done

if grep -q "completed VOS prediction on 155 videos" "$LORA_LOG" 2>/dev/null; then
    log_event "LoRA experiment is already complete; nothing to do."
    exit 0
fi

log_event "Baseline complete; starting SA-V LoRA experiment."
env MPLCONFIGDIR=/tmp/lit-matplotlib "$PYTHON_BIN" -u "$EVAL_SCRIPT" \
    --base_video_dir "$DATA_DIR/JPEGImages_24fps" \
    --input_mask_dir "$DATA_DIR/Annotations_6fps" \
    --video_list_file "$VIDEO_LIST" \
    --output_mask_dir "$LORA_OUTPUT" \
    --per_obj_png_file \
    --no-track_object_appearing_later_in_video \
    --online_evaluation_unlimited \
    --correct_threshold 0.5 \
    --LIT_LoRA_mode \
    > "$LORA_LOG" 2>&1
lora_status=$?
log_event "SA-V LoRA experiment exited with status $lora_status."
exit "$lora_status"
