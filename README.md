# Branch for AgentDojo Experiment

## Setup Environment
After setup the environment from the main branch, you just need to install AgentDojo:

```
pip install -e AgentDojo
```

This AgentDojo is the exact same version of the original one, but we have added the support to Azure GPT-4o, and attack type (SuperDirectAttack) that just returns the injection goal for training that we replace the injection goal with the adversarial prompt.

## Usage
### Prepare Data
You can run `data/AgentDojo/split_dataset.py`to split the dataset.

### Training
You can use similar training script in the main branch, but just change the dataset path and reward function. For example:

```
# Add your API keys here
export AZURE_API_VERSION=2024-06-01
export GPT_4O_AZURE_API_VERSION=2024-06-01
export GPT_4O_API_KEY=XXX
export GPT_4O_ENDPOINT=XXX

export WANDB_PROJECT="RL-Hammer"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LR=1e-5
RUN_NAME=rl_hammer_agentdojo_target_gpt_4o_lora

ATTACKER_MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct
TARGET_MODEL_NAME_OR_PATH=gpt-4o

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch \
    train.py \
    --attacker_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
    --target_model_name_or_path ${TARGET_MODEL_NAME_OR_PATH} \
    --target_model_url dummy \
    --reward_functions AgentDojoReward \
    --dataset data/AgentDojo/dataset/train.json \
    --attn_implementation flash_attention_2 \
    --num_generations 32 \
    --num_iterations 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 20 \
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
        python agentdojo_eval.py \
            --attacker_model_name_or_path ${dir} \
            --attacker_base_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
            --target_model_name_or_path ${TARGET_MODEL_NAME_OR_PATH} \
            --validation_data_path data/AgentDojo/dataset/eval.json \
            --enable_wandb True \
            --run_name agentdojo_eval_${RUN_NAME}_attack_gpt-4o
    fi
done

```

### Evaluation
Evaluate baseline injections:
```
# Tool Knowledge
export CUDA_VISIBLE_DEVICES=0
python agentdojo_eval.py \
    --attacker_model_name_or_path tool_knowledge \
    --target_model_name_or_path gpt-4o \
    --validation_data_path data/AgentDojo/dataset/test.json \
    --enable_wandb True \
    --run_name agentdojo_eval_tool_knowledge_attack_gpt-4o
```

Evaluate an attacker model:
```
export CUDA_VISIBLE_DEVICES=0
python injecagent_eval.py \
    --attacker_model_name_or_path ${CHECKPOINT} \
    --attacker_base_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --target_model_name_or_path gpt-4o \
    --validation_data_path data/AgentDojo/dataset/test.json \
    --enable_wandb True \
    --run_name agentdojo_eval_${RUN_NAME}_attack_gpt-4o
```

## License
The majority of code is under [CC-BY-NC 4.0 license](LICENSE). TRL library is available under [Apache-2.0 License](https://github.com/huggingface/trl). InjecAgent is under [MIT License](https://github.com/uiuc-kang-lab/InjecAgent/tree/main).
