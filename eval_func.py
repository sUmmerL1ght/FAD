import time

from PIL import Image
from tqdm import tqdm
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion import Diffusion
from models import Transformer, Critic, LSTMPolicy, RNNPolicy, GRUPolicy
from temporal import TemporalUnet, UncondTemporalUnet
from datasets.normalization import DatasetNormalizer
from Flow_policy import FlowMatching, FlowMatching_euler, FlowMatching_unet_euler, FlowMatching_expand_mlp_euler, FlowMatching_expand_mlp_euler_OT
from Flow_policy import TrajectoryFlowNetwork, MLP_expand

torch.backends.cudnn.enabled = True

class Eval_Policy(object):
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
        self.history = history
        if history is None:
            self.history = horizon - 1
        input_dim = state_dim
        self.min_action = float(action_scale.low[0])
        self.max_action = float(action_scale.high[0])
        self.mask = None

        print('use model: ', model)
        if model == 'transformer':
            n_heads, n_layers = 4, 4
            self.feasible_generator = Transformer(input_dim, d_model, n_heads, d_ff, n_layers, 0.1).to(device)
        elif model == 'lstm':
            self.feasible_generator = LSTMPolicy(state_dim, length=horizon).to(device)
        elif model == 'rnn':
            self.feasible_generator = RNNPolicy(state_dim, length=horizon).to(device)
        elif model == 'gru':
            self.feasible_generator = GRUPolicy(state_dim, length=horizon).to(device)
        elif model == 'flow':
            self.flow_model = UncondTemporalUnet(horizon, state_dim, attention=use_attention, dim_mults=dim_mults).to(
                device)
            self.feasible_generator = FlowMatching(state_dim, model=self.flow_model, device=device, horizon=horizon).to(
                device)
        elif model == 'flow_cond':
            self.flow_model = TemporalUnet(horizon, state_dim, None, attention=use_attention, dim_mults=dim_mults).to(device)
            self.feasible_generator = FlowMatching(state_dim, model=self.flow_model, device=device, horizon=horizon).to(device)
        elif model == 'flow_mini_unet':
            self.flow_model = UncondTemporalUnet(horizon, state_dim, dim=32, attention=use_attention,
                                                 dim_mults=(1, 2, 4)).to(device)
            self.feasible_generator = FlowMatching(state_dim, model=self.flow_model, device=device, horizon=horizon).to(
                device)
        elif model == 'flow_mlp_euler':
            self.flow_model = TrajectoryFlowNetwork(horizon, state_dim, hidden_dim=256).to(device)
            self.feasible_generator = FlowMatching_euler(state_dim, model=self.flow_model, device=device,
                                                         horizon=horizon).to(
                device)
        elif model == 'flow_unet_euler':
            self.flow_model = UncondTemporalUnet(horizon, state_dim, attention=use_attention, dim_mults=dim_mults).to(
                device)
            self.feasible_generator = FlowMatching_unet_euler(state_dim, model=self.flow_model, device=device,
                                                              horizon=horizon).to(
                device)
        elif model == 'expand_mlp_euler':
            self.flow_model = MLP_expand(horizon * state_dim, w=128, time_emb_dim=32).to(device)
            self.feasible_generator = FlowMatching_expand_mlp_euler(state_dim, model=self.flow_model, device=device,
                                                                    horizon=horizon).to(
                device)
        elif model == 'expand_mlp_euler_OT':
            self.flow_model = MLP_expand(horizon * state_dim, w=128, time_emb_dim=32).to(device)
            self.feasible_generator = FlowMatching_expand_mlp_euler_OT(state_dim, model=self.flow_model, device=device,
                                                                       horizon=horizon).to(
                device)
        self.feasible_generator_optimizer = torch.optim.Adam(self.feasible_generator.parameters(), lr=lr)

        # =====================================================================#
        # ============================= Diffuser ==============================#
        # =====================================================================#
        self.model = TemporalUnet(horizon, state_dim, None, attention=use_attention, dim_mults=dim_mults).to(device)
        self.planner = Diffusion(state_dim, self.model, None, horizon=horizon, n_timesteps=n_timesteps,
                                 predict_epsilon=True, beta_schedule=schedule, w=w).to(device)  # predict_epsilon=False
        self.planner_optimizer = torch.optim.AdamW(self.planner.parameters(), lr=lr, weight_decay=1e-4)

        self.ema_model = copy.deepcopy(self.planner)
        self.ema_model2 = copy.deepcopy(self.feasible_generator)  # ema_flow

        self.update_ema_every = 2


        # =====================================================================#
        # =============================== Actor ===============================#
        # =====================================================================#
        hidden_dim = 256
        self.actor = nn.Sequential(
            nn.Linear(2 * self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # =====================================================================#
        # =============================== Critic ==============================#
        # =====================================================================#
        self.critic = Critic(state_dim, action_dim, length=horizon).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.device = device
        self.discount = discount
        self.tau = tau
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_timesteps = n_timesteps

        self.planner_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.planner_optimizer, 1, 1e-2, total_iters=1e5)
        self.feasible_generator_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.feasible_generator_optimizer, 1,
                                                                                 1e-2, total_iters=1e5)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, 1, 1e-2, total_iters=1e5)

        self.sqrt_alpha = self.planner.sqrt_alphas_cumprod[n_timesteps - 1].unsqueeze(-1)
        self.sqrt_one_minus_alphas_cumprod = self.planner.sqrt_one_minus_alphas_cumprod[n_timesteps - 1].unsqueeze(-1)

        self.times = 0
        self.steps = 0

    @torch.no_grad()
    def evaluate(self, env, eval_episodes=10, normalizer: DatasetNormalizer = None, rtg=None, scale=1.0,
                 use_diffusion=True, progress=False, use_ddim=False):
        scores = []
        # self.feasible_generator.eval()
        # self.ema_model2.eval()  # ema_flow
        # self.ema_model.eval()

        # render os
        current_file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(current_file_path)
        image_path = os.path.join(dir_path, f"image/{env}-FAD/render_img")
        os.makedirs(image_path, exist_ok=True)

        return_to_go = rtg
        # from utils.rendering import MuJoCoRenderer
        # render = MuJoCoRenderer(self.env_name)
        for _ in range(eval_episodes):
            state, done = env.reset(), False

            # frame = env.render(mode="rgb_array")  # mujoco maze2d
            frame = env.sim.render(width=640, height=480, camera_name="fixed") # adort (pen:fixed) (door:vil_camera) (hammer:vil_camera) (relocate:vil_camera)
            img = Image.fromarray(frame)
            img.save(os.path.join(image_path, f'frame-{0}.png'))

            history_state = []
            history_rtg = []
            episode_reward = 0
            rtg = return_to_go


            rtg = rtg / scale
            min_rtg = rtg / 3
            total_step = 0

            while not done:
                if isinstance(state, dict):
                    state = state["observation"]
                state = normalizer.normalize(state, "observations")
                history_state.append(state)
                history_rtg.append(rtg)
                # queue: np.ndarray => torch [1, horizon, state_dim]
                _state: torch.Tensor = torch.tensor(np.stack(history_state, 0), dtype=torch.float32).unsqueeze(0).to(
                    self.device)
                _rtg: torch.Tensor = torch.tensor(np.stack(history_rtg, 0), dtype=torch.float32).unsqueeze(0).to(
                    self.device)

                time1 = time.time()

                action = self.select_action(_state, _rtg, use_diffusion, False, normalizer)

                t = time.time() - time1
                self.times += t
                self.steps += 1
                if self.steps > 1000:
                    print(self.times, self.steps)

                state, reward, done, _ = env.step(action)

                if self.steps % 10 == 0:
                # if 1:
                    # frame = env.render(mode="rgb_array")  # mujoco maze2d
                    frame = env.sim.render(width=640, height=480, camera_name="fixed") # adort (pen:fixed) (door:vil_camera) (hammer:vil_camera) (relocate:vil_camera)
                    img = Image.fromarray(frame)
                    img.save(os.path.join(image_path, f'frame-{total_step}.png'))

                if done:
                    pass
                rtg -= reward / scale

                rtg = np.clip(rtg, min_rtg, None)
                episode_reward += reward
                if 1:
                    total_step += 1
                    # print(f"                                                              ", end="\r")
                    # print(f"steps: {total_step} -------- rewards: {episode_reward}", end="\r")
            # unnormaled_state = normalizer.unnormalize(_state[:1].detach().cpu().data.numpy(), "observations")
            # render.composite(f"./reference/{self.env_name}_traj.png", unnormaled_state)
            if progress:
                print(
                    f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}")
            scores.append(episode_reward)
        # self.feasible_generator.train()
        # self.ema_model2.train()  # ema_flow
        # self.ema_model.train()

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)

        min_score = np.min(scores)
        print(min_score)

        # normlize
        normalized_scores = [env.get_normalized_score(s) for s in scores]
        avg_normalized_score = env.get_normalized_score(avg_score)
        std_normalized_score = np.std(normalized_scores)
        max_normalized_score = np.max(normalized_scores)

        return {"reward/avg": avg_score,
                "reward/std": std_score,
                "reward/avg_normalized": avg_normalized_score,
                "reward/std_normalized": std_normalized_score,
                "reward/max": max_score,
                "reward/max_normalized": max_normalized_score}

    def load(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        checkpoint = torch.load(path, map_location=self.device)
        self.planner.load_state_dict(checkpoint['planner'])
        self.feasible_generator.load_state_dict(checkpoint['flow_matching'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def select_action(self, state, rtg, use_diffusion=True, use_ddim=False, normalizer=None):
        repeat = 32
        state = torch.repeat_interleave(state, repeat, dim=0)
        rtg = torch.repeat_interleave(rtg, repeat, dim=0).unsqueeze(-1)
        cond = {0: state[:, -1]}
        condtion = rtg[:, -1]
        # from utils.rendering import MuJoCoRenderer
        # render = MuJoCoRenderer(self.env_name)
        with torch.no_grad():
            if not use_ddim:
                # state_flow = self.feasible_generator(repeat, cond=cond)
                # state_flow = self.ema_model2(repeat, cond=cond)  # ema_flow
                state_flow = self.feasible_generator(repeat, cond=cond, reward=condtion)  # ema_flow_cond

                if use_diffusion:
                    _noise_state = state_flow
                    state = self.planner(_noise_state, cond, condtion)
                    # if length % 100 == 0:
                    # state = torch.cat([_noise_state, _state], dim=0)
            else:
                # self.ema_model.n_timesteps = 10
                state = torch.randn(state.shape[0], self.horizon, self.state_dim).to(self.device)
                state = self.ema_model.ddim_sample(state, cond, condtion, ddim_timesteps=10)
                # unnormaled_state = normalizer.unnormalize(state[:1].detach().cpu().data.numpy(), "observations")
                # render.composite(f"./reference/{self.env_name}_sample-reference.png", unnormaled_state)

            _cond = state[:, :2, :]
            # assert not torch.isnan(_cond).any(), f"state: {_cond}"
            actions = self.actor(_cond.contiguous().view(-1, self.state_dim * 2)).squeeze()
            # assert not torch.isnan(actions).any(), f"actions: {actions}"
            reward = self.critic(_cond[:, 0], actions).flatten()
            # not inf nan
            # assert not torch.isnan(reward).any(), f"reward: {reward}"
            idx = torch.argmax(reward)
            # idx = torch.multinomial(F.softmax(reward, dim=0), num_samples=1)
            action = actions[idx].squeeze()
        return action.cpu().data.numpy().flatten()

    # def calculate(self, env_name, eval_episodes=10, normalizer: DatasetNormalizer = None, rtg=None, scale=1.0,
    #               show_progress=True, seed=None):
    #     print('Evaluate seed: ', seed)
    #     set_seed(seed)
    #     env = gym.make(env_name)
    #
    #     return_to_go = rtg
    #     scores = []
    #     reward_list = [[] for _ in range(eval_episodes)]
    #
    #     for idx in range(eval_episodes):
    #         self.break_count = 0
    #         state, done = env.reset(
    #             seed=seed), False  # directly input the seed into reset to ensure that the initial state is fixed
    #
    #         history_state = []
    #         history_rtg = []
    #         episode_reward = 0
    #         rtg = return_to_go
    #         rtg = rtg / scale
    #         min_rtg = rtg / 3
    #         total_step = 0
    #         break_flag = False
    #
    #         if isinstance(state, dict):
    #             state = state["observation"]  # shape:(state_dim,)
    #         state = normalizer.normalize(state, "observations")
    #         history_state.append(state)
    #         history_rtg.append(rtg)
    #
    #         while not done:
    #             _state: torch.Tensor = torch.tensor(np.stack(history_state, 0), dtype=torch.float32,
    #                                                 device=self.device).unsqueeze(0)
    #             _rtg: torch.Tensor = torch.tensor(np.stack(history_rtg, 0), dtype=torch.float32,
    #                                               device=self.device).unsqueeze(0)
    #
    #             action_list, threshold_list = self.select_action(_state, _rtg)
    #             for j in range(7):
    #                 action = action_list[j]
    #                 state, reward, done, _ = env.step(action)
    #                 rtg -= reward / scale
    #                 reward_list[idx].append(reward)
    #                 if self.use_adapt_rs:
    #                     break_flag = (rtg < self.threshold_scale * threshold_list[j])
    #                 rtg = np.clip(rtg, min_rtg, None)
    #                 episode_reward += reward
    #
    #                 if isinstance(state, dict):
    #                     state = state["observation"]  # shape:(state_dim,)
    #                 state = normalizer.normalize(state, "observations")
    #                 history_state.append(state)
    #                 history_rtg.append(rtg)
    #
    #                 if show_progress:
    #                     total_step += 1
    #                     print(f"                                                              ", end="\r")
    #                     print(f"steps: {total_step} -------- rewards: {episode_reward}", end="\r")
    #
    #                 if done: break
    #
    #                 if self.use_adapt_rs and break_flag:
    #                     self.break_count += 1
    #                     break
    #
    #         if show_progress:
    #             print(
    #                 f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}, break_count: {self.break_count}")
    #
    #         scores.append(episode_reward)
    #
    #     idx = scores.index(max(scores))
    #     max_score = max(scores)
    #
    #     print("max:", np.round(max_score, 4))
    #     print("max list:", np.round(np.array(reward_list[idx]), 4).tolist())