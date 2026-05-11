from tqdm import tqdm
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion import Diffusion
from models import Transformer, Critic, LSTMPolicy, RNNPolicy, GRUPolicy, ValueTransformer
from temporal import TemporalUnet, UncondTemporalUnet
from datasets.normalization import DatasetNormalizer
from Flow_policy import FlowMatching, FlowMatching_euler, FlowMatching_unet_euler, FlowMatching_expand_mlp_euler, FlowMatching_expand_mlp_euler_OT
from Flow_policy import TrajectoryFlowNetwork, MLP_expand

torch.backends.cudnn.enabled = True


def generate_square_subsequent_mask(seq: torch.Tensor):
    sz_b, len_s, *_ = seq.shape
    return torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1).bool()


class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_average(ma_params.data, current_params.data)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def _build_flow_model(model_type, horizon, state_dim, use_attention, dim_mults, d_model, d_ff, device):
    if model_type == 'transformer':
        n_heads, n_layers = 4, 4
        flow_model = None
        feasible_generator = Transformer(state_dim, d_model, n_heads, d_ff, n_layers, 0.1).to(device)
    elif model_type == 'flow':
        flow_model = UncondTemporalUnet(horizon, state_dim, attention=use_attention, dim_mults=dim_mults).to(device)
        feasible_generator = FlowMatching(state_dim, model=flow_model, device=device, horizon=horizon).to(device)
    elif model_type == 'flow_cond':
        flow_model = TemporalUnet(horizon, state_dim, None, attention=use_attention, dim_mults=dim_mults).to(device)
        feasible_generator = FlowMatching(state_dim, model=flow_model, device=device, horizon=horizon).to(device)
    elif model_type == 'flow_mini_unet':
        flow_model = UncondTemporalUnet(horizon, state_dim, dim=32, attention=use_attention, dim_mults=(1, 2, 4)).to(device)
        feasible_generator = FlowMatching(state_dim, model=flow_model, device=device, horizon=horizon).to(device)
    elif model_type == 'flow_mlp_euler':
        flow_model = TrajectoryFlowNetwork(horizon, state_dim, hidden_dim=256).to(device)
        feasible_generator = FlowMatching_euler(state_dim, model=flow_model, device=device, horizon=horizon).to(device)
    elif model_type == 'flow_unet_euler':
        flow_model = UncondTemporalUnet(horizon, state_dim, attention=use_attention, dim_mults=dim_mults).to(device)
        feasible_generator = FlowMatching_unet_euler(state_dim, model=flow_model, device=device, horizon=horizon).to(device)
    elif model_type == 'expand_mlp_euler':
        flow_model = MLP_expand(horizon * state_dim, w=128, time_emb_dim=32).to(device)
        feasible_generator = FlowMatching_expand_mlp_euler(state_dim, model=flow_model, device=device, horizon=horizon).to(device)
    elif model_type == 'expand_mlp_euler_OT':
        flow_model = MLP_expand(horizon * state_dim, w=128, time_emb_dim=32).to(device)
        feasible_generator = FlowMatching_expand_mlp_euler_OT(state_dim, model=flow_model, device=device, horizon=horizon).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return feasible_generator


class Policy_fm(object):
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
                 model='flow',
                 use_inner_critic=False):
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
        self.use_inner_critic = use_inner_critic

        self.feasible_generator = _build_flow_model(model, horizon, state_dim, use_attention, dim_mults, d_model, d_ff, device)
        self.feasible_generator_optimizer = torch.optim.Adam(self.feasible_generator.parameters(), lr=lr)

        self.model = TemporalUnet(horizon, state_dim, None, attention=use_attention, dim_mults=dim_mults).to(device)
        self.planner = Diffusion(state_dim, self.model, None, horizon=horizon, n_timesteps=n_timesteps,
                                 predict_epsilon=True, beta_schedule=schedule, w=w).to(device)
        self.planner_optimizer = torch.optim.AdamW(self.planner.parameters(), lr=lr, weight_decay=1e-4)

        self.ema = EMA(1 - tau)
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

        if use_inner_critic:
            n_heads, n_layers = 4, 4
            self.inner_critic = ValueTransformer(state_dim, action_dim, d_model, n_heads, d_ff, n_layers).to(device)
            self.inner_critic_optimizer = torch.optim.Adam(self.inner_critic.parameters(), lr=lr)

        self.planner_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.planner_optimizer, 1, 1e-2, total_iters=1e5)
        self.feasible_generator_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.feasible_generator_optimizer, 1, 1e-2, total_iters=1e5)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, 1, 1e-2, total_iters=1e5)

        self.sqrt_alpha = self.planner.sqrt_alphas_cumprod[n_timesteps - 1].unsqueeze(-1)
        self.sqrt_one_minus_alphas_cumprod = self.planner.sqrt_one_minus_alphas_cumprod[n_timesteps - 1].unsqueeze(-1)

    def eval(self):
        self.feasible_generator.eval()
        self.ema_model.eval()
        self.ema_model2.eval()
        self.planner.eval()
        self.actor.eval()
        self.critic.eval()

    def evaluate(self, env, eval_episodes=10, normalizer: DatasetNormalizer = None, rtg=None, scale=1.0,
                 use_diffusion=True, progress=False, use_ddim=False):
        scores = []
        self.ema_model2.eval()
        self.ema_model.eval()

        return_to_go = rtg
        for _ in range(eval_episodes):
            state, done = env.reset(), False
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
                action = self.select_action(_state, _rtg, use_diffusion, False, normalizer)
                state, reward, done, _ = env.step(action)
                rtg = np.clip(rtg - reward / scale, min_rtg, None)
                episode_reward += reward
                if progress:
                    total_step += 1
                    print(f"steps: {total_step} -------- rewards: {episode_reward}", end="\r")
            if progress:
                print(f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}")
            scores.append(episode_reward)

        self.ema_model2.train()
        self.ema_model.train()

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

    def train(self, batch):
        observations, next_observations, actions, rtg = batch
        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)
        actions = actions.to(self.device)
        rtg_raw = rtg.to(self.device)
        rtg = rtg.unsqueeze(-1).to(self.device)

        cond = {0: observations[:, 0]}

        t_loss = self.feasible_generator.loss(observations, cond, rtg[:, 0])
        self.feasible_generator_optimizer.zero_grad()
        t_loss.backward()
        self.feasible_generator_optimizer.step()

        p_loss = self.planner.loss(observations, cond, rtg[:, 0])
        self.planner_optimizer.zero_grad()
        p_loss.backward()
        self.planner_optimizer.step()

        s_ns_pair = torch.cat([observations, next_observations], dim=-1).view(-1, self.state_dim * 2)
        actions_flat = actions.view(-1, self.action_dim)
        pred_actions = self.actor(s_ns_pair).clamp(self.min_action, self.max_action)
        a_loss = F.mse_loss(pred_actions, actions_flat)
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        value = self.critic(observations.reshape(-1, self.state_dim), actions_flat)
        c_loss = F.mse_loss(value, rtg.contiguous().view(-1, 1))
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()

        losses = {
            "loss/planner": p_loss.item(),
            "loss/flow": t_loss.item(),
            "loss/critic": c_loss.item(),
            "loss/actor": a_loss.item(),
        }

        if self.use_inner_critic:
            inner_value = self.inner_critic(observations, actions)
            inner_c_loss = F.mse_loss(inner_value, rtg_raw.unsqueeze(-1))
            self.inner_critic_optimizer.zero_grad()
            inner_c_loss.backward()
            self.inner_critic_optimizer.step()
            losses["loss/inner_critic"] = inner_c_loss.item()

        self.planner_lr_scheduler.step()
        self.feasible_generator_lr_scheduler.step()
        self.actor_lr_scheduler.step()
        return losses

    def save(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            'planner': self.planner.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'flow_matching': self.feasible_generator.state_dict(),
            'ema_model_flow': self.ema_model2.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }
        if self.use_inner_critic:
            ckpt['inner_critic'] = self.inner_critic.state_dict()
        torch.save(ckpt, path)

    def load(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        checkpoint = torch.load(path, map_location=self.device)
        self.planner.load_state_dict(checkpoint['planner'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.feasible_generator.load_state_dict(checkpoint['flow_matching'])
        self.ema_model2.load_state_dict(checkpoint['ema_model_flow'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if self.use_inner_critic:
            self.inner_critic.load_state_dict(checkpoint['inner_critic'])

    @torch.no_grad()
    def select_action(self, state, rtg, use_diffusion=True, use_ddim=False, normalizer=None):
        repeat = 32
        state = torch.repeat_interleave(state, repeat, dim=0)
        rtg = torch.repeat_interleave(rtg, repeat, dim=0).unsqueeze(-1)
        cond = {0: state[:, -1]}
        condition = rtg[:, -1]

        if not use_ddim:
            state_flow = self.ema_model2(repeat, cond=cond, reward=condition)
            if use_diffusion:
                state = self.ema_model(state_flow, cond, condition)
            else:
                state = state_flow
        else:
            state = torch.randn(state.shape[0], self.horizon, self.state_dim).to(self.device)
            state = self.ema_model.ddim_sample(state, cond, condition, ddim_timesteps=10)

        _cond = state[:, :2, :]
        actions = self.actor(_cond.contiguous().view(-1, self.state_dim * 2)).squeeze()
        reward = self.critic(_cond[:, 0], actions).flatten()
        idx = torch.multinomial(F.softmax(reward, dim=0), num_samples=1)
        return actions[idx].squeeze().cpu().data.numpy().flatten()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.planner.state_dict())
        self.ema_model2.load_state_dict(self.feasible_generator.state_dict())

    def step_ema(self, step, step_start_ema):
        if step < step_start_ema:
            self.reset_parameters()
            return
        if step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.planner)
            self.ema.update_model_average(self.ema_model2, self.feasible_generator)


class Policy_fm_doublecritic(Policy_fm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_inner_critic=True)
