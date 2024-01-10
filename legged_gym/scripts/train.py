# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import sys
sys.path.append("/home/g54/issac_gym")
sys.path.append("/home/g54/issac_gym/legged_gym")
sys.path.append("/home/g54/issac_gym/rsl_rl")
import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import json
def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def record_param(info,dir):
    reward_scales = class_to_dict( info.rewards.scales)
    commands = class_to_dict( info.commands)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    json_file_path = "/home/g54/issac_gym/legged_gym/logs/"+dir+"/myDictionary.json"
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as file:
            json.dump([], file)
    try:
        # 以读写模式打开（'r+'）文件，如果文件不存在，则以写模式（'w+'）创建文件
        with open(json_file_path, 'r+') as file:
            # 读取文件中的内容
            try:
                file_data = json.load(file)
                # 检查文件内容是否为列表，如果不是，创建一个列表
                if not isinstance(file_data, list):
                    file_data = []
            except json.JSONDecodeError:
                # 如果文件是空的或内容不是JSON格式，则创建一个空列表
                file_data = []
            
            # 将新的时间和内容添加到列表中
            #file_data.append({'experiment name': dir})
            file_data.append({'experiment name': dir,'time': current_time, 'reward_scales': reward_scales,'commamd': commands})
            
            # 移动到文件开头
            file.seek(0)
            
            # 将更新后的数据写回文件
            json.dump(file_data, file, indent=4)
    except IOError as e:
        print(f"An error occurred: {e}")
     
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    record_param(env_cfg,train_cfg.runner.experiment_name)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
