


pred_mask = (
    out_mask_logits[0] > score_thresh
).cpu().numpy().reshape(height, width)


# =====================================================
# 1. AFR：主动判断当前帧是否失败
# =====================================================

failure_result = predictor.detect_failure(
    inference_state=inference_state,
    frame_idx=out_frame_idx,
    obj_id=object_id,
    pred_mask=pred_mask,
    reference=R0,
)

if failure_result["is_failure"]:

    failure_cause = failure_result["failure_cause"]

    # 当前失败经验
    experience = build_failure_experience(
        frame_idx=out_frame_idx,
        failure_result=failure_result,
        pred_mask=pred_mask,
    )


    # =================================================
    # 2. MGA：检索同类历史经验
    # =================================================

    retrieved_experiences = retrieve_memory(
        semantic_memory,
        failure_cause=failure_cause,
        current_experience=experience,
    )


    # =================================================
    # 3. 生成纠正 prompt
    # =================================================

    correction_prompt = generate_correction_prompt(
        reference=R0,
        current_experience=experience,
        retrieved_experiences=retrieved_experiences,
    )


    # =================================================
    # 4. SAM3重新得到纠正后的mask
    # =================================================

    correct_result = predictor.correct_by_semantic_prompt(
        inference_state=inference_state,
        frame_idx=out_frame_idx,
        obj_id=object_id,
        correction_prompt=correction_prompt,
    )

    corr_mask = correct_result["mask"]


    # =================================================
    # 5. Difference supervision
    # =================================================

    diff_mask = np.logical_xor(
        pred_mask,
        corr_mask
    ).astype(np.float32)


    # =================================================
    # 6. LoRA在线更新
    # =================================================

    if not trained_lora:

        mask_decoder_lora = copy.deepcopy(
            predictor.sam_mask_decoder
        )

        predictor.convert_to_lora(
            mask_decoder_lora.transformer
        )

        predictor.freeze_non_lora(
            mask_decoder_lora
        )

        predictor.train_lora(
            mask_decoder_lora,
            corr_mask,
            diff_mask=diff_mask,
            training_epoch=training_epoch,
        )

        trained_lora = True

    else:

        predictor.train_lora(
            predictor.LIT_lora,
            corr_mask,
            diff_mask=diff_mask,
            training_epoch=training_epoch,
            mode="finetune",
        )


    # =================================================
    # 7. 当前经验写入 Semantic Memory
    # =================================================

    semantic_memory.append(experience)


    # 使用纠正后的结果
    video_segments[out_frame_idx][object_id] = corr_mask