# RL Is a Hammer and LLMs Are Nails: A Simple Reinforcement Learning Recipe for Strong Prompt Injection

This code is the official implementation of [RL-Hammer](https://arxiv.org/abs/2510.04885).

## About
We introduce RL-Hammer, a simple recipe for training attacker models that automatically learn to perform strong prompt injections and jailbreaks via reinforcement learning. RL-Hammer requires no warm-up data and can be trained entirely from scratch. To achieve high ASRs against industrial-level models with defenses, we propose a set of practical techniques that enable highly effective, universal attacks.

## Setup Environment

```
conda create -n rl-hammer python=3.12 -y
conda activate rl-hammer

pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation --no-cache-dir
```

## Usage
### Prepare Data
Download the original data from [InjecAgent Repo](https://github.com/uiuc-kang-lab/InjecAgent/blob/main/data/test_cases_dh_base.json), and move `test_cases_dh_base.json` to `data/InjecAgent/raw`. Then you can run `python data/InjecAgent/split_dataset.py`to split the dataset.

Second, download the [tool file](https://github.com/uiuc-kang-lab/InjecAgent/blob/main/data/tools.json) and move it to `data/InjecAgent`.

### Merge Meta-SecAlign
Run `python merge_meta_secalign.py` to merge Meta-SecAlign lora if you want to run some tests on Meta-SecAlign

### Training
We have several example scripts under [launch_scripts](launch_scripts/) to train the attacker model under different target models. Make sure you add your API keys in the script.

You can find the training scripts for experiments using the diversity reward in [diversity_reward_scripts](launch_scripts/diversity_reward_scripts).

Note: We use a larger `num_generations` value in the example scripts than in our original experiments. Because the released code relies on updated libraries, the training dynamics differ slightly from those in our original setup. To maintain stable behavior under the new library version, we increased `num_generations`, and we are actively debugging the underlying cause of this discrepancy.

Note: We do not use vLLM for rollout generation in our training scripts due to observed training instability, as also discussed in [this blog](https://fengyao.notion.site/off-policy-rl).

### Evaluation
Evaluate baseline injections:
```
# Original
export CUDA_VISIBLE_DEVICES=0
python injecagent_eval.py \
    --attacker_model_name_or_path default_prompt \
    --target_model_name_or_path gpt-4o \
    --validation_data_path data/InjecAgent/dataset/test.json \
    --enable_wandb True \
    --run_name eval_default_prompt_attack_gpt-4o

# Enhanced
export CUDA_VISIBLE_DEVICES=0
python injecagent_eval.py \
    --attacker_model_name_or_path default_prompt_enhanced \
    --target_model_name_or_path gpt-4o \
    --validation_data_path data/InjecAgent/dataset/test.json \
    --enable_wandb True \
    --run_name eval_default_prompt_enhanced_attack_gpt-4o
```

Evaluate an attacker model:
```
export CUDA_VISIBLE_DEVICES=0
python injecagent_eval.py \
    --attacker_model_name_or_path ${CHECKPOINT} \
    --attacker_base_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --target_model_name_or_path gpt-4o \
    --validation_data_path data/InjecAgent/dataset/test.json \
    --enable_wandb True \
    --run_name eval_${RUN_NAME}_attack_gpt-4o
```

## License
The majority of code is under [CC-BY-NC 4.0 license](LICENSE). TRL library is available under [Apache-2.0 License](https://github.com/huggingface/trl). InjecAgent is under [MIT License](https://github.com/uiuc-kang-lab/InjecAgent/tree/main).
