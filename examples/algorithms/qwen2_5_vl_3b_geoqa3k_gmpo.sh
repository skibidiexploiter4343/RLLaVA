#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

LOCAL_DATASET_PATH="hiyouga/geometry3k"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@test"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=qwen2_5_vl_3b_geoqa3k_gmpo

torchrun --nproc_per_node=2 -m rllava.train.pipeline.rlvr \
    config=examples/config.yaml \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.val_batch_size=1000 \
    data.format_prompt=./examples/format_prompt/math.jinja \
    actor.model.model_path=${MODEL_PATH} \
    actor.clip_ratio_low=0.4 \
    actor.clip_ratio_high=0.4 \
    actor.policy_loss.loss_mode=geo_mean \
    actor.use_kl_loss=false \
    reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR}
