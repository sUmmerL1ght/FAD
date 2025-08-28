import os
import random

import gym
import d4rl
import json
import heapq
import torch
import argparse
import numpy as np
from eval_func import Eval_Policy
from eval_multi_func import Eval_Multi_Policy
from eval_maze import Eval_Maze
from datasets.dataset import SequenceDatasetV2
from datasets.normalization import DatasetNormalizer
import datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

hyperparameters = {
    'halfcheetah-medium-expert-v2': {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 12000.0}, # 1.1,12000
    'halfcheetah-medium-replay-v2': {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0}, # 1.1,5300
    'halfcheetah-medium-v2': {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 6000.0}, # 1.1,5300
    'hopper-medium-expert-v2': {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3600.0}, # 16
    'hopper-medium-replay-v2': {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0}, # 16
    'hopper-medium-v2': {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0},  # n_step=5
    'walker2d-medium-expert-v2': {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 55, 'scalar': 1.1, 'rtg': 6000.0}, # 5100
    'walker2d-medium-replay-v2': {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 6000.0}, # 4200
    'walker2d-medium-v2': {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 6000.0}, # 4200
    'pen-human-v1': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0},
    'pen-cloned-v1': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0},
    'pen-expert-v1': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0},
    'kitchen-partial-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 500.0},
    'kitchen-mixed-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 400.0},
    'door-human-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 1500.0},
    'door-cloned-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 1500.0},
    'door-expert-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 1500.0},
    'hammer-human-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 17000.0},
    'hammer-cloned-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 17000.0},
    'hammer-expert-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 17000.0},
    'relocate-human-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 4000.0},
    'relocate-cloned-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 4000.0},
    'relocate-expert-v0': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 4000.0},
    'maze2d-umaze-v1': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 200.0},
    'maze2d-medium-v1': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 300.0},
    'maze2d-large-v1': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 400.0},
    'antmaze-umaze-v2': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0},
    'antmaze-large-play-v2': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0},
    'antmaze-large-diverse-v2': {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0},
    'FetchPush-v1': {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0,
                     'load_path': '/data3/hrenming/Trajectory_Diffuser/data/hard_tasks_2e5/expert_small/FetchPush'},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='pen-expert-v1')
    parser.add_argument('--seed', type=int, default=712)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=5e4)
    parser.add_argument('--save_freq', type=int, default=2e5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--model', type=str, default='flow_cond')  # flow
    return parser.parse_args()




def main():
    args = parse_args()
    gamma = args.gamma
    schedule = args.schedule
    eval_freq = args.eval_freq
    env_name = args.env_name
    tau = args.tau
    model = args.model
    seed = args.seed

    horizon = hyperparameters[env_name]['horizon']
    n_timesteps = hyperparameters[env_name]['n_timesteps']
    lr = hyperparameters[env_name]['lr']
    w = hyperparameters[env_name]['scalar']
    rtg = hyperparameters[env_name]['rtg']

    try:
        load_path = hyperparameters[env_name]['load_path']
    except:
        load_path = None

    # env = datasets.load_environment(env_name)  # for maze
    env = gym.make(env_name)  # for others
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


    if 'Fetch' in env_name:
        observation_dim = env.observation_space['observation'].shape[0]
    else:
        observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space

    scale = 1000
    if 'maze2d' in args.env_name:
        scale = 500
    elif 'halfcheetah' in args.env_name:
        scale = 10000
    elif 'antmaze' in args.env_name:
        scale = 1
    elif 'kitchen' in args.env_name:
        scale = 100

    policy_eval = Eval_Policy(args.env_name,
                       observation_dim,
                       action_dim,
                       action_scale,
                       horizon,
                       device,
                       w=w,
                       discount=gamma,
                       tau=tau,
                       n_timesteps=n_timesteps,
                       lr=lr,
                       schedule=schedule,
                       model=model)

    dataset = SequenceDatasetV2(env_name,
                                horizon=horizon,
                                returns_scale=scale,
                                termination_penalty=None,
                                load_path=load_path)
    normalizer: DatasetNormalizer = dataset.normalizer

    cnt = 600000

    policy_eval.load(f"./models/{env_name}_{model}_{seed}/{cnt}.pth")

    print("=================EVAL START=================")
    rew_record = []
    for i in range(1):
        set_seed(i*42+34)
        env.seed(i*42+34)
        reward_td = policy_eval.evaluate(env, 1, normalizer, rtg, scale, True)
        rew_record.append(reward_td['reward/avg_normalized'])
        formate_print(reward_td, i)

    avg = np.mean(rew_record)
    std_normalized_score = np.std(rew_record)
    max_normalized_score = np.max(rew_record)
    print("avg:", avg)
    print("std:", std_normalized_score)
    print("max:", max_normalized_score)


def formate_print(reward: dict, cnt):
    print("========================================")
    print(f"times: {cnt}")
    print("----------------------------------------")
    for key in reward:
        print(f"{key}: {reward[key]}")
    print("========================================")


def record(writer, prefix, scalar, gloabl_step):
    for key in scalar:
        writer.add_scalar(f"{prefix}/{key}", scalar[key], gloabl_step)




if __name__ == "__main__":
    main()
