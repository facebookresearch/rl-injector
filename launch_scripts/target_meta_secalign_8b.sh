#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

export WANDB_PROJECT="RL-Hammer"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LR=1e-5
RUN_NAME=rl_hammer_target_meta_secalign_8b_lora

ATTACKER_MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct

# Launch target model on GPU 7
export CUDA_VISIBLE_DEVICES=7
python -m vllm.entrypoints.openai.api_server \
    --model checkpoints/Meta-SecAlign-8B-merged \
    --port 8010 \
    --gpu_memory_utilization 0.45 > /dev/null 2>&1 &

until curl -s http://localhost:8010/v1/models; do sleep 5; done

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8011 \
    --gpu_memory_utilization 0.45 > /dev/null 2>&1 &

until curl -s http://localhost:8011/v1/models; do sleep 5; done

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
accelerate launch \
    train.py \
    --attacker_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
    --target_model_name_or_path "checkpoints/Meta-SecAlign-8B-merged;meta-llama/Llama-3.1-8B-Instruct" \
    --target_model_url "http://localhost:8010/v1;http://localhost:8011/v1" \
    --reward_functions InjecAgentToolCallingReward \
    --dataset data/InjecAgent/dataset/train.json \
    --attn_implementation flash_attention_2 \
    --num_generations 8 \
    --num_iterations 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 40 \
    --bf16 True \
    --beta 0.0 \
    --warmup_ratio 0.03 \
    --gradient_checkpointing True \
    --learning_rate ${LR} \
    --lr_scheduler_type constant_with_warmup \
    --use_peft True \
    --lora_r 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir checkpoints/${RUN_NAME} \
    --report_to wandb \
    --run_name ${RUN_NAME}

# Eval all checkpoints
export CUDA_VISIBLE_DEVICES=0
for dir in checkpoints/${RUN_NAME}/*; do
    if [ -d "$dir" ]; then
        python injecagent_eval.py \
            --attacker_model_name_or_path ${dir} \
            --attacker_base_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
            --target_model_name_or_path checkpoints/Meta-SecAlign-8B-merged \
            --validation_data_path data/InjecAgent/dataset/eval.json \
            --enable_wandb True \
            --run_name eval_${RUN_NAME}_attack_meta_secalign_8b
    fi
done
