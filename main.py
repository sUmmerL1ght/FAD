from typing import Dict, List, Tuple, Union
import os
import shutil
import argparse
import tqdm
import gym
import numpy as np
import torch
import d4rl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from policy_fm import Policy_fm, Policy_fm_doublecritic
from datasets.dataset import SequenceDatasetV2
from datasets.normalization import DatasetNormalizer
from helpers import cycle
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


hyperparameters = {
    # Gym-MuJoCo locomotion
    'halfcheetah-medium-expert-v2': {'lr': 3e-4, 'horizon': 8,  'n_timesteps': 5, 'scalar': 1.1, 'rtg': 12000.0, 'scale': 10000},
    'halfcheetah-medium-replay-v2': {'lr': 3e-4, 'horizon': 8,  'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0,  'scale': 10000},
    'halfcheetah-medium-v2':        {'lr': 3e-4, 'horizon': 8,  'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0,  'scale': 10000},
    'hopper-medium-expert-v2':      {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3600.0,  'scale': 1000},
    'hopper-medium-replay-v2':      {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0,  'scale': 1000},
    'hopper-medium-v2':             {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0,  'scale': 1000},
    'walker2d-medium-expert-v2':    {'lr': 3e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5100.0,  'scale': 1000},
    'walker2d-medium-replay-v2':    {'lr': 3e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 4200.0,  'scale': 1000},
    'walker2d-medium-v2':           {'lr': 3e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 4200.0,  'scale': 1000},
    # Maze2D
    'maze2d-umaze-v1':  {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 200.0, 'scale': 500},
    'maze2d-medium-v1': {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 300.0, 'scale': 500},
    'maze2d-large-v1':  {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 400.0, 'scale': 500},
    # AntMaze
    'antmaze-umaze-v2':        {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0, 'scale': 1},
    'antmaze-medium-play-v2':  {'lr': 3e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0, 'scale': 1},
    'antmaze-medium-diverse-v2': {'lr': 3e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0, 'scale': 1},
    'antmaze-large-play-v2':   {'lr': 3e-4, 'horizon': 64, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0, 'scale': 1},
    'antmaze-large-diverse-v2': {'lr': 3e-4, 'horizon': 64, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0, 'scale': 1},
    # Adroit (Pen)
    'pen-human-v1':  {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0, 'scale': 1000},
    'pen-cloned-v1': {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0, 'scale': 1000},
    'pen-expert-v1': {'lr': 3e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0, 'scale': 1000},
    # Kitchen
    'kitchen-partial-v0': {'lr': 3e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 500.0, 'scale': 100},
    'kitchen-mixed-v0':   {'lr': 3e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 400.0, 'scale': 100},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1')
    parser.add_argument('--seed', type=int, default=0)
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
    save_freq = args.save_freq
    env_name = args.env_name
    tau = args.tau
    model = args.model
    seed = args.seed
    set_seed(seed)

    horizon = hyperparameters[env_name]['horizon']
    n_timesteps = hyperparameters[env_name]['n_timesteps']
    lr = hyperparameters[env_name]['lr']
    w = hyperparameters[env_name]['scalar']
    rtg = hyperparameters[env_name]['rtg']
    env = gym.make(env_name)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space

    scale = hyperparameters[env_name]['scale']

    policy = Policy_fm_doublecritic(args.env_name,
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
                                termination_penalty=None)
    normalizer: DatasetNormalizer = dataset.normalizer
    step_start_ema = 10000
    cnt = 0
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    totle_iteration = 1_000_000
    if os.path.exists(f"./runs/{env_name}_{model}_{seed}"):
        shutil.rmtree(f"./runs/{env_name}_{model}_{seed}")

    writer = SummaryWriter(f"./runs/{env_name}_{model}_{seed}")

    print("=================TRAINING START=================")
    for _, batch in enumerate(cycle(dataloader)):
        cnt += 1
        loss: Dict = policy.train(batch)
        policy.step_ema(cnt, step_start_ema)

        record(writer, "loss", loss, cnt)
        if cnt % eval_freq == 0:
            reward_td = policy.evaluate(env, 5, normalizer, rtg, scale, True)
            record(writer, "reward_td", reward_td, cnt)
            formate_print(loss, reward_td, cnt)
        if cnt == 1 or cnt % save_freq == 0:
            policy.save(f"./models/{env_name}_{model}_{seed}/{cnt}.pth")
        if cnt == totle_iteration:
            break


def formate_print(loss: dict, reward: dict, cnt):
    print("========================================")
    print(f"iteration: {cnt}")
    for key in loss:
        print(f"{key}: {loss[key]}")
    print("----------------------------------------")
    for key in reward:
        print(f"{key}: {reward[key]}")
    print("========================================")


def record(writer, prefix, scalar, gloabl_step):
    for key in scalar:
        writer.add_scalar(f"{prefix}/{key}", scalar[key], gloabl_step)


if __name__ == "__main__":
    main()