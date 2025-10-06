# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig


@dataclass
class LocalGRPOConfig(GRPOConfig):
    dataset: str = field(default=None)
    epsilon: float = field(default=0.5)

    attacker_model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.1-8B-Instruct"
    )
    model_dtype: Optional[str] = field(default="bfloat16")
    target_model_name_or_path: Optional[str] = field(default="")
    max_completion_length: int = field(default=1024)
    save_total_limit: int = field(default=10)
    seed: int = field(default=1024)
    max_grad_norm: float = field(default=0.2)
    loss_type: str = field(default="bnpo")

    reward_functions: list[str] = field(
        default_factory=lambda: ["InjecAgentToolCallingReward"]
    )
    target_model_url: str = field(default="http://localhost:8000/v1")
    soft_rewards: bool = field(default=True)
    target_model_max_completion_length: int = field(default=512)
    target_model_temperature: float = field(default=None)
    reasoning_effort: str = field(default="minimal")  # minimal, low, medium, high
    model_wise_reward_weights: Optional[list[float]] = field(default=None)


@dataclass
class EvalConfig:
    validation_data_path: str = None
    val_batch_size: int = 16
    max_new_tokens: int = 1024
    val_max_new_tokens: int = 512

    attacker_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    attacker_base_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    target_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    reasoning_effort: str = field(default="minimal")  # minimal, low, medium, high
    attacker_model_dtype: str = "bfloat16"
    target_model_dtype: str = "bfloat16"
    temperature: float = None

    enable_wandb: bool = False
    wandb_project_name: str = "RL-Hammer"
    run_name: str = "test"
    output_dir: str = "outputs/test"
    save_name: str = "default"
