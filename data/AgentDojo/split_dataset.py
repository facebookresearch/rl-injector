# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os
import itertools

from agentdojo.scripts.benchmark import get_suites, get_suite

benchmark_version = "v1.2.1"

all_suites = tuple(get_suites(benchmark_version).keys())

data = []
for suite_name in all_suites:
    suite = get_suite(benchmark_version, suite_name)
    all_tasks = list(suite.user_tasks.keys())
    all_injections = list(suite.injection_tasks)

    for task_name, injection_name in list(itertools.product(all_tasks, all_injections)):
        data.append(
            {
                "suite_name": suite_name,
                "task_name": task_name,
                "injection_name": injection_name,
            }
        )

random.shuffle(data)

# 100 for eval; 100 for test; rest for training
train_data = data[:-200]
eval_data = data[-200:-100]
test_data = data[-100:]

os.makedirs("data/AgentDojo/dataset", exist_ok=True)
with open("data/AgentDojo/dataset/train.json", "w") as file:
    json.dump(train_data, file, indent=4)
with open("data/AgentDojo/dataset/eval.json", "w") as file:
    json.dump(eval_data, file, indent=4)
with open("data/AgentDojo/dataset/test.json", "w") as file:
    json.dump(test_data, file, indent=4)
