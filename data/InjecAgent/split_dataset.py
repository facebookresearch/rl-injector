# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os

with open("data/InjecAgent/raw/test_cases_dh_base.json", "r") as file:
    data = json.load(file)

random.shuffle(data)

# 100 for eval; 100 for test; rest for training
train_data = data[:-200]
eval_data = data[-200:-100]
test_data = data[-100:]

os.makedirs("data/InjecAgent/dataset", exist_ok=True)
with open("data/InjecAgent/dataset/train.json", "w") as file:
    json.dump(train_data, file, indent=4)
with open("data/InjecAgent/dataset/eval.json", "w") as file:
    json.dump(eval_data, file, indent=4)
with open("data/InjecAgent/dataset/test.json", "w") as file:
    json.dump(test_data, file, indent=4)
