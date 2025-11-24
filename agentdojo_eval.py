# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoTokenizer, HfArgumentParser
from datasets import Dataset
from torch.utils.data import DataLoader
from vllm import LLM
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from vllm.lora.request import LoRARequest
import contextlib
import gc
import ray
import json
import copy
from tqdm import tqdm
import wandb

from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.agent_pipeline.agent_pipeline import (
    AgentPipeline,
    PipelineConfig,
)
from agentdojo.scripts.benchmark import load_attack, get_suite, get_suites

from config import EvalConfig
from utils import ATTACKER_SYS_PROMPT
from reward_func import extract_attack_prompt, AgentDojoReward


def delete_vllm_model(model):
    destroy_model_parallel()
    destroy_distributed_environment()
    model.llm_engine.engine_core.shutdown()
    del model
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


def main():
    parser = HfArgumentParser(EvalConfig)
    args = parser.parse_args_into_dataclasses()[0]

    if args.enable_wandb is True:
        wandb.init(project=args.wandb_project_name, name=args.run_name)

        wandb_table = wandb.Table(
            columns=[
                "adv_goal",
                "attacker_output",
                "attacker_adv_prompt",
                "agentdojo_output",
                "if_attack_success",
            ]
        )

    validation_data = json.load(open(args.validation_data_path, "r"))

    # Load all necessary stuff for AgentDojo
    all_agentdojo_attacks = list(ATTACKS.keys())
    benchmark_version = "v1.2.1"
    all_suite_names = tuple(get_suites(benchmark_version).keys())
    all_suites = {}
    all_user_tasks = {}
    all_injections = {}
    for suite_name in all_suite_names:
        curr_suite = get_suite(benchmark_version, suite_name)
        all_suites[suite_name] = curr_suite
        all_user_tasks[suite_name] = curr_suite.user_tasks
        all_injections[suite_name] = curr_suite.injection_tasks

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=args.target_model_name_or_path,
            defense=None,
            system_message_name=None,
            system_message=None,
            tool_output_format=None,
        )
    )

    if args.attacker_model_name_or_path in all_agentdojo_attacks:
        # Perform AgentDojo built-in attacks
        adv_prompt_results = []
        for i in range(len(validation_data)):
            curr_data_row = validation_data[i]
            suite_name = curr_data_row["suite_name"]
            injection_name = curr_data_row["injection_name"]
            task_name = curr_data_row["task_name"]

            curr_suite = copy.deepcopy(all_suites[suite_name])
            curr_injection = copy.deepcopy(all_injections[suite_name][injection_name])
            curr_task = copy.deepcopy(all_user_tasks[suite_name][task_name])
            curr_attacker = load_attack(
                args.attacker_model_name_or_path, curr_suite, pipeline
            )
            attack_result = curr_attacker.attack(curr_task, curr_injection)
            curr_adv_prompt = list(attack_result.values())[0]

            adv_prompt_results.append(
                {
                    "adv_goal": curr_injection.GOAL,
                    "attacker_output": "",
                    "attacker_adv_prompt": curr_adv_prompt,
                }
            )
    else:
        for i in range(len(validation_data)):
            curr_data_row = validation_data[i]
            suite_name = curr_data_row["suite_name"]
            injection_name = curr_data_row["injection_name"]
            task_name = curr_data_row["task_name"]

            curr_suite = copy.deepcopy(all_suites[suite_name])
            curr_injection = copy.deepcopy(all_injections[suite_name][injection_name])
            curr_task = copy.deepcopy(all_user_tasks[suite_name][task_name])
            curr_attacker = load_attack("tool_knowledge", curr_suite, pipeline)
            attack_result = curr_attacker.attack(curr_task, curr_injection)
            curr_adv_prompt = list(attack_result.values())[0]
            validation_data[i]["Attacker Instruction"] = curr_adv_prompt

        # Load dataset
        validation_dataset = Dataset.from_list(validation_data)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
        )

        # Perform attack
        if "lora" in args.attacker_model_name_or_path.lower():
            attacker_model = LLM(
                model=args.attacker_base_model_name_or_path,
                dtype=args.attacker_model_dtype,
                trust_remote_code=True,
                enable_lora=True,
                max_lora_rank=128,
            )
            lora_request = LoRARequest(
                "attack_lora", 1, lora_path=args.attacker_model_name_or_path
            )
        else:
            attacker_model = LLM(
                model=args.attacker_model_name_or_path,
                dtype=args.attacker_model_dtype,
                trust_remote_code=True,
            )
            lora_request = None
        attacker_tokenizer = AutoTokenizer.from_pretrained(
            args.attacker_model_name_or_path, trust_remote_code=True
        )

        adv_prompt_results = []
        for validation_step, validation_batch in tqdm(
            enumerate(validation_loader),
            total=len(validation_loader),
            desc="Generating adversarial prompts",
        ):
            # Generate adversarial prompt
            attacker_goals = validation_batch["Attacker Instruction"]
            attacker_prompts = [
                ATTACKER_SYS_PROMPT.format(goal=attacker_goal)
                for attacker_goal in attacker_goals
            ]
            attacker_messages = [
                [{"role": "user", "content": attacker_prompt}]
                for attacker_prompt in attacker_prompts
            ]
            attacker_input_texts = attacker_tokenizer.apply_chat_template(
                attacker_messages, add_generation_prompt=True, tokenize=False
            )

            sampling_params = attacker_model.get_default_sampling_params()
            if args.temperature is not None:
                sampling_params.temperature = args.temperature
            sampling_params.max_tokens = args.max_new_tokens
            attacker_outputs = attacker_model.generate(
                attacker_input_texts, sampling_params, lora_request=lora_request
            )
            attacker_output_texts = [
                output.outputs[0].text for output in attacker_outputs
            ]

            # Extract the attack prompt from the output
            for i in range(len(validation_batch["Attacker Instruction"])):
                attacker_output_text = attacker_output_texts[i]
                attacker_goal = validation_batch["Attacker Instruction"][i]

                # Extract the attack prompt from the output
                attacker_adv_prompt = extract_attack_prompt(attacker_output_text)

                # For yaml format
                attacker_adv_prompt = attacker_adv_prompt.replace('"', "")
                attacker_adv_prompt = attacker_adv_prompt.replace("\\", "")
                attacker_adv_prompt = attacker_adv_prompt.replace("---", "- - -")
                attacker_adv_prompt = attacker_adv_prompt.replace("...", ". . .")

                adv_prompt_results.append(
                    {
                        "adv_goal": attacker_goal,
                        "attacker_output": attacker_output_text,
                        "attacker_adv_prompt": attacker_adv_prompt,
                    }
                )

        # Delete the attacker model to free up memory
        delete_vllm_model(attacker_model)
        del attacker_tokenizer

    # Add sample_id
    for i in range(len(adv_prompt_results)):
        adv_prompt_results[i]["sample_id"] = i

    # Make adv_prompt_results a dataset and dataloader
    adv_prompt_results_dataset = Dataset.from_list(adv_prompt_results)
    adv_prompt_results_dataloader = DataLoader(
        adv_prompt_results_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
    )

    agentdojo_reward_func = AgentDojoReward(config=args)
    target_model_results = []
    for adv_prompt_batch in tqdm(
        adv_prompt_results_dataloader, desc="Attacking target model"
    ):
        all_inputs = []
        for i in range(len(adv_prompt_batch["adv_goal"])):
            curr_data_row = validation_data[adv_prompt_batch["sample_id"][i]]
            curr_suite = all_suites[curr_data_row["suite_name"]]
            curr_task = all_user_tasks[curr_data_row["suite_name"]][
                curr_data_row["task_name"]
            ]
            curr_injection = all_injections[curr_data_row["suite_name"]][
                curr_data_row["injection_name"]
            ]

            curr_suite = copy.deepcopy(curr_suite)
            curr_task = copy.deepcopy(curr_task)
            curr_injection = copy.deepcopy(curr_injection)

            # Overwrite the goal
            curr_injection.GOAL = adv_prompt_batch["attacker_adv_prompt"][i]
            dummy_attack = load_attack("superdirect", curr_suite, pipeline)
            task_injections = dummy_attack.attack(curr_task, curr_injection)
            all_inputs.append(
                [
                    curr_suite,
                    curr_task,
                    curr_injection,
                    task_injections,
                ]
            )

        # Run AgentDojo
        all_completions = agentdojo_reward_func.run_agentdojo_batched(
            all_inputs, pipeline
        )

        # Extract results
        for i in range(len(adv_prompt_batch["adv_goal"])):
            target_model_output = (
                f"Utility: {all_completions[i][0]}, Security: {all_completions[i][1]}"
            )
            target_model_results.append(
                {
                    "adv_goal": adv_prompt_batch["adv_goal"][i],
                    "attacker_output": adv_prompt_batch["attacker_output"][i],
                    "attacker_adv_prompt": adv_prompt_batch["attacker_adv_prompt"][i],
                    "target_model_output": target_model_output,
                }
            )

    # Judge
    attack_success_count = 0
    utility_count = 0
    final_results = []
    for i in range(len(target_model_results)):
        curr_row = target_model_results[i]
        curr_data_row = validation_data[i]

        if_attack_success = "Security: True" in curr_row["target_model_output"]

        final_results.append(
            {
                "adv_goal": curr_row["adv_goal"],
                "attacker_output": curr_row["attacker_output"],
                "attacker_adv_prompt": curr_row["attacker_adv_prompt"],
                "target_model_output": curr_row["target_model_output"],
            }
        )

        if if_attack_success is True:
            attack_success_count += 1

        if "Utility: True" in curr_row["target_model_output"]:
            utility_count += 1

        # Log the results
        if args.enable_wandb is True:
            wandb_table.add_data(
                curr_row["adv_goal"],
                curr_row["attacker_output"],
                curr_row["attacker_adv_prompt"],
                curr_row["target_model_output"],
                if_attack_success,
            )

    attack_success_rate = attack_success_count / len(final_results)
    print(f"Validation completed. Attack success rate: {attack_success_rate:.2%}")
    utility_success_rate = utility_count / len(final_results)

    if args.enable_wandb is True:
        wandb.log({"attack_success_rate": attack_success_rate})
        wandb.log({"utility_success_rate": utility_success_rate})
        wandb.log({"validation_table": wandb_table})
        wandb.finish()


if __name__ == "__main__":
    main()
