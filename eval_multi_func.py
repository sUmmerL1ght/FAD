import time
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion import Diffusion
from models import Critic, ValueTransformer
from temporal import TemporalUnet
from datasets.normalization import DatasetNormalizer
from policy_fm import _build_flow_model
from PIL import Image

torch.backends.cudnn.enabled = True


class Eval_Multi_Policy(object):
    def __init__(self,
                 env_name,
                 state_dim,
                 action_dim,
                 action_scale,
                 horizon,
                 device,
                 discount,
                 tau,
                 n_timesteps,
                 d_model=256,
                 d_ff=512,
                 use_attention=False,
                 dim_mults=(1, 2, 4, 8),
                 w=1,
                 history=None,
                 schedule="linear",
                 lr=3e-3,
                 model='flow') -> None:

        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history = horizon - 1 if history is None else history
        self.min_action = float(action_scale.low[0])
        self.max_action = float(action_scale.high[0])
        self.device = device
        self.discount = discount
        self.tau = tau
        self.horizon = horizon
        self.n_timesteps = n_timesteps

        self.feasible_generator = _build_flow_model(model, horizon, state_dim, use_attention, dim_mults, d_model, d_ff, device)
        self.feasible_generator_optimizer = torch.optim.Adam(self.feasible_generator.parameters(), lr=lr)

        self.model = TemporalUnet(horizon, state_dim, None, attention=use_attention, dim_mults=dim_mults).to(device)
        self.planner = Diffusion(state_dim, self.model, None, horizon=horizon, n_timesteps=n_timesteps,
                                 predict_epsilon=True, beta_schedule=schedule, w=w).to(device)
        self.planner_optimizer = torch.optim.AdamW(self.planner.parameters(), lr=lr, weight_decay=1e-4)

        self.ema_model = copy.deepcopy(self.planner)
        self.ema_model2 = copy.deepcopy(self.feasible_generator)
        self.update_ema_every = 2

        hidden_dim = 256
        self.actor = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, length=horizon).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        n_heads, n_layers = 4, 4
        self.inner_critic = ValueTransformer(state_dim, action_dim, d_model, n_heads, d_ff, n_layers).to(device)
        self.inner_critic_optimizer = torch.optim.Adam(self.inner_critic.parameters(), lr=lr)

        self.planner_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.planner_optimizer, 1, 1e-2, total_iters=1e5)
        self.feasible_generator_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.feasible_generator_optimizer, 1, 1e-2, total_iters=1e5)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, 1, 1e-2, total_iters=1e5)

        self.sqrt_alpha = self.planner.sqrt_alphas_cumprod[n_timesteps - 1].unsqueeze(-1)
        self.sqrt_one_minus_alphas_cumprod = self.planner.sqrt_one_minus_alphas_cumprod[n_timesteps - 1].unsqueeze(-1)

        self.steps = 0

    @torch.no_grad()
    def evaluate(self, env, eval_episodes=10, normalizer: DatasetNormalizer = None, rtg=None, scale=1.0,
                 use_diffusion=True, progress=False, use_ddim=False):
        scores = []

        current_file_path = os.path.abspath(__file__)
        image_path = os.path.join(os.path.dirname(current_file_path), f"image/{env}-FAD/render_img")
        os.makedirs(image_path, exist_ok=True)

        return_to_go = rtg
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            img = Image.fromarray(env.render(mode="rgb_array"))
            img.save(os.path.join(image_path, f'frame-{0}.png'))

            history_state, history_rtg = [], []
            episode_reward = 0
            rtg = return_to_go / scale
            min_rtg = rtg / 3
            total_step = 0

            while not done:
                if isinstance(state, dict):
                    state = state["observation"]
                state = normalizer.normalize(state, "observations")
                history_state.append(state)
                history_rtg.append(rtg)
                _state = torch.tensor(np.stack(history_state, 0), dtype=torch.float32).unsqueeze(0).to(self.device)
                _rtg = torch.tensor(np.stack(history_rtg, 0), dtype=torch.float32).unsqueeze(0).to(self.device)

                action_list = self.select_action(_state, _rtg, use_diffusion, False, normalizer)

                for i in range(len(action_list)):
                    state, reward, done, _ = env.step(action_list[i])
                    self.steps += 1
                    if self.steps % 100 == 0:
                        img = Image.fromarray(env.render(mode="rgb_array"))
                        img.save(os.path.join(image_path, f'frame-{total_step}.png'))
                    rtg = np.clip(rtg - reward / scale, min_rtg, None)
                    episode_reward += reward
                    total_step += 1

            if progress:
                print(f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}")
            scores.append(episode_reward)

        avg_score = np.mean(scores)
        normalized_scores = [env.get_normalized_score(s) for s in scores]
        return {
            "reward/avg": avg_score,
            "reward/std": np.std(scores),
            "reward/avg_normalized": env.get_normalized_score(avg_score),
            "reward/std_normalized": np.std(normalized_scores),
            "reward/max": np.max(scores),
            "reward/max_normalized": np.max(normalized_scores),
        }

    def load(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        checkpoint = torch.load(path, map_location=self.device)
        self.planner.load_state_dict(checkpoint['planner'])
        self.feasible_generator.load_state_dict(checkpoint['flow_matching'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.inner_critic.load_state_dict(checkpoint['inner_critic'])

    @torch.no_grad()
    def select_action(self, state, rtg, use_diffusion=True, use_ddim=False, normalizer=None):
        repeat = 32
        state = torch.repeat_interleave(state, repeat, dim=0)
        rtg = torch.repeat_interleave(rtg, repeat, dim=0).unsqueeze(-1)
        cond = {0: state[:, -1]}
        condition = rtg[:, -1]

        if not use_ddim:
            state_flow = self.feasible_generator(repeat, cond=cond, reward=condition)
            if use_diffusion:
                state = self.planner(state_flow, cond, condition)
            else:
                state = state_flow
        else:
            state = torch.randn(state.shape[0], self.horizon, self.state_dim).to(self.device)
            state = self.ema_model.ddim_sample(state, cond, condition, ddim_timesteps=10)

        _cond = state[:, :2, :]
        actions = self.actor(_cond.contiguous().view(-1, self.state_dim * 2)).squeeze()
        reward = self.critic(_cond[:, 0], actions).flatten()
        idx = torch.multinomial(F.softmax(reward, dim=0), num_samples=1)

        action_list = []
        action_list_c = []
        action_list_c.append(actions[idx].squeeze())
        action_list.append(action_list_c[0].cpu().data.numpy().flatten())

        h = state.shape[1]
        for ii in range(h - 2):
            _cond_inner = state[:, ii+1:ii+3, :]
            actions_ = self.actor(_cond_inner.contiguous().view(-1, self.state_dim * 2)).squeeze()
            action__c = actions_[idx].squeeze()
            action_list.append(action__c.cpu().data.numpy().flatten())
            action_list_c.append(action__c)

        action_list_c = torch.stack(action_list_c, dim=0)
        value = self.inner_critic(state[idx, :-1, :], action_list_c.unsqueeze(0)).squeeze(0)

        action_num = -1
        for i in range(h - 3):
            if value[i+1] < value[i+2]:
                action_num = i + 1
                break

        return action_list[:action_num]
