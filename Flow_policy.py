import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq

# import flow-based
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.optimal_transport import OTPlanSampler

import numpy as np
from typing import Dict, List, Tuple, Union


ot_sampler = OTPlanSampler(method='exact')


# OT
def OT_sample(x0, x1):
    x0, x1 = ot_sampler.sample_plan(x0, x1)
    return x0, x1

# condition
def apply_condition(seq, cond):
    for key, value in cond.items():
        seq[:, key] = value.clone()
    return seq

# unet_neuralODE
class FlowMatching(nn.Module):
    def __init__(
            self,
            state_dim: int,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            model = None,
            horizon: int = 100,
            device: str = "cuda"
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def compute_conditional_vector_field(self, x0, x1):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """

        return x1 - x0

    def sample_conditional_pt(self, x0, x1, t, sigma):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon

    def forward(self, bs, num_steps=1, cond=None, reward=None) -> torch.Tensor:
    # def forward(self, bs, num_steps=20, cond=None) -> torch.Tensor:
        x_ini = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x_ini = apply_condition(x_ini, cond)


        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, cond=reward, time=torch.ones(bs).to(self.device) * t, use_dropout=False),
            x_ini,
            torch.linspace(0, 1, num_steps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )

        # traj = torchdiffeq.odeint(
        #     lambda t, x: self.model.forward(x, t),
        #     x_ini,
        #     torch.linspace(0, 1, num_steps + 1).to(self.device),
        #     atol=1e-4,
        #     rtol=1e-4,
        #     method="euler",
        # )

        # node = NeuralODE(
        #     torch_wrapper(self.model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        # )
        #
        # with torch.no_grad():
        #     traj = node.trajectory(
        #         x_ini,
        #         t_span=torch.linspace(0, 1, num_steps),
        #     )

        return traj[-1]

    def loss(self, x: torch.Tensor, cond, reward) -> torch.Tensor:
    # def loss(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        # x0, x = OT_sample(x0, x)  # OT_sample
        if cond is not None:
            x0 = apply_condition(x0, cond)
        # x0, x = OT_sample(x0, x)  # OT_sample
        xt = self.sample_conditional_pt(x0, x, t, sigma=0.01)
        # if cond is not None:
        #     xt = apply_condition(xt, cond)  # need?
        ut = self.compute_conditional_vector_field(x0, x)
        # vt = self.model(xt, t)
        vt = self.model(xt, reward, t)  # cond
        loss = F.mse_loss(vt, ut)
        return loss

# mlp for expand
class MLP_expand(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_emb_dim=16):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        # 时间编码层 - 将1维时间扩展为多维
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 主网络 - 拼接了扩展后的时间表示
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + time_emb_dim, w),  # 输入是状态+扩展时间
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, t):
        # 扩展时间维度
        t = t.view(-1, 1)  # 调整形状为 [batch_size, 1]
        t_emb = self.time_encoder(t)  # [batch_size, time_emb_dim]

        # 拼接扩展后的时间和状态
        inputs = torch.cat([x, t_emb], dim=1)

        return self.net(inputs)


# position encoding for rnn_mlp
class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦余弦位置编码

    参数:
        d_model: 编码维度
        max_len: 最大序列长度 (默认=5000)
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 使用正弦函数进行偶数位置编码
        pe[:, 0::2] = torch.sin(position * div_term)

        # 使用余弦函数进行奇数位置编码
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加批次维度并注册为缓冲区（不作为模型参数）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        输入:
            x: 可以是以下类型:
                - 形状 [batch_size] 的张量 (单个时间步/位置)
                - 形状 [batch_size, seq_len] 的张量 (序列)
                - 形状 [batch_size, seq_len, *] 的张量 (带特征的序列)
        输出:
            位置编码, 形状为 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        if x.dim() == 1:
            # 单个时间步/位置
            x = x.unsqueeze(1)  # [batch_size, 1]
            return self.pe[:, x.long(), :].squeeze(1)
        elif x.dim() == 2:
            # 序列 [batch_size, seq_len]
            return self.pe[:, :x.size(1), :]
        else:
            # 带特征的序列，我们只需要位置信息
            return self.pe[:, :x.size(1), :]



#  rnn_mlp network model
class TrajectoryFlowNetwork(nn.Module):
    def __init__(self, seq_len, feature_dim, hidden_dim=256):
        super().__init__()

        # 时间编码
        self.time_encoder = nn.Sequential(
            SinusoidalPositionalEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 特征编码器 (维持序列结构)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 时序编码 - 使用双向LSTM或Transformer来捕获序列内关联
        self.sequence_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # 速度场预测器
        self.velocity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x, t):
        """
        输入:
            x: 轨迹数据, 形状为 [batch_size, seq_len, feature_dim]
            t: 时间点, 形状为 [batch_size]
        输出:
            velocity: 预测的速度场, 形状为 [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, feature_dim = x.shape

        # 时间编码
        t_emb = self.time_encoder(t)  # [batch_size, hidden_dim]
        t_emb = t_emb.squeeze(0).squeeze(1)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]

        # 特征编码 (应用于每个时间步)
        x_emb = self.feature_encoder(x)  # [batch_size, seq_len, hidden_dim]

        # 合并时间和特征编码
        combined = x_emb + t_emb  # [batch_size, seq_len, hidden_dim]

        # 序列编码
        seq_features, _ = self.sequence_encoder(combined)  # [batch_size, seq_len, hidden_dim]

        # 合并所有信息并预测速度场
        velocity_input = torch.cat([seq_features, combined], dim=-1)  # [batch_size, seq_len, hidden_dim*2]
        velocity = self.velocity_predictor(velocity_input)  # [batch_size, seq_len, feature_dim]

        return velocity


# euler method
def generate_trajectory(model, batch_size, seq_len, feature_dim, device, cond=None, steps=100, method='euler'):
    """
    使用训练好的Flow Matching模型生成轨迹

    参数:
        model: 训练好的速度场模型
        batch_size: 生成的轨迹批次大小
        seq_len: 轨迹序列长度
        feature_dim: 轨迹特征维度
        device: 计算设备
        steps: 积分步数
        method: 积分方法 ('euler', 'rk4')

    返回:
        生成的轨迹数据, 形状为 [batch_size, seq_len, feature_dim]
    """
    # 初始化随机噪声
    x = torch.randn(batch_size, seq_len, feature_dim).to(device)
    if cond is not None:
        x = apply_condition(x, cond)

    # 定义时间步长
    dt = 1.0 / steps

    # 执行数值积分
    # model.eval()
    with torch.no_grad():
        for i in range(steps):
            t = torch.ones(batch_size).to(device) * i * dt

            if method == 'euler':
                # Euler方法
                velocity = model(x, t)
                x = x + velocity * dt
            elif method == 'rk4':
                # 四阶Runge-Kutta方法
                k1 = model(x, t)
                k2 = model(x + k1 * dt / 2, t + dt / 2)
                k3 = model(x + k2 * dt / 2, t + dt / 2)
                k4 = model(x + k3 * dt, t + dt)
                x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    return x

#  rnn_mlp_euler
class FlowMatching_euler(nn.Module):
    def __init__(
            self,
            state_dim: int,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            model = None,
            horizon: int = 100,
            device: str = "cuda"
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_conditional_pt(self, x0, x1, t, sigma):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon

    def forward(self, bs, num_steps=100, cond=None) -> torch.Tensor:
        pred_x = generate_trajectory(self.model, bs, self.horizon, self.state_dim, self.device, cond, num_steps)

        return pred_x

    def loss(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        if cond is not None:
            x = apply_condition(x, cond)
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        xt = self.sample_conditional_pt(x0, x, t, sigma=0.01)
        ut = self.compute_conditional_vector_field(x0, x)
        vt = self.model(xt, t)
        loss = F.mse_loss(vt, ut)
        return loss


#  unet_euler
class FlowMatching_unet_euler(nn.Module):
    def __init__(
            self,
            state_dim: int,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            model = None,
            horizon: int = 100,
            device: str = "cuda"
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def compute_conditional_vector_field(self, x0, x1):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """

        return x1 - x0

    def sample_conditional_pt(self, x0, x1, t, sigma):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon

    def forward(self, bs, num_steps=100, cond=None) -> torch.Tensor:
        x = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x = apply_condition(x, cond)

        # 定义时间步长
        dt = 1.0 / num_steps

        # 执行数值积分
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.ones(bs).to(self.device) * i * dt
                velocity = self.model(x, t)
                x = x + velocity * dt
                # if cond is not None:
                #     x = apply_condition(x, cond)

        return x

    def loss(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        if cond is not None:
            x0 = apply_condition(x0, cond)
        x0, x = OT_sample(x0, x)
        xt = self.sample_conditional_pt(x0, x, t, sigma=0.01)
        # if cond is not None:
        #     xt = apply_condition(xt, cond)
        ut = self.compute_conditional_vector_field(x0, x)
        vt = self.model(xt, t)
        loss = F.mse_loss(vt, ut)
        return loss


#  expand_mlp_euler
class FlowMatching_expand_mlp_euler(nn.Module):
    def __init__(
            self,
            state_dim: int,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            model = None,
            horizon: int = 100,
            device: str = "cuda"
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def expand_x(self, x):
        x = x.reshape(x.shape[0], -1)
        return x

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_conditional_pt(self, x0, x1, t, sigma):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon

    def forward(self, bs, num_steps=100, cond=None) -> torch.Tensor:
        x = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x = apply_condition(x, cond)
        x = self.expand_x(x)

        # 定义时间步长
        dt = 1.0 / num_steps

        # 执行数值积分
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.ones(bs).to(self.device) * i * dt
                velocity = self.model(x, t)
                x = x + velocity * dt

        x = x.reshape(bs, self.horizon, self.state_dim)

        return x

    def loss(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        if cond is not None:
            x = apply_condition(x, cond)
        batch_size = x.shape[0]
        x = self.expand_x(x)
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        xt = self.sample_conditional_pt(x0, x, t, sigma=0.01)
        ut = self.compute_conditional_vector_field(x0, x)
        vt = self.model(xt, t)
        loss = F.mse_loss(vt, ut)
        return loss

#  expand_mlp_euler with OT
class FlowMatching_expand_mlp_euler_OT(nn.Module):
    def __init__(
            self,
            state_dim: int,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            model = None,
            horizon: int = 100,
            device: str = "cuda"
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def expand_x(self, x):
        x = x.reshape(x.shape[0], -1)
        return x

    def invers_expand_x(self, x):
        x = x.reshape(x.shape[0], self.horizon, -1)
        return x

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_conditional_pt(self, x0, x1, t, sigma):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon

    def forward(self, bs, num_steps=100, cond=None) -> torch.Tensor:
        x = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x = apply_condition(x, cond)
        x = self.expand_x(x)

        # 定义时间步长
        dt = 1.0 / num_steps

        # 执行数值积分
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.ones(bs).to(self.device) * i * dt
                velocity = self.model(x, t)
                x = x + velocity * dt
                x = self.invers_expand_x(x)
                if cond is not None:
                    x = apply_condition(x, cond)
                x = self.expand_x(x)

        x = x.reshape(bs, self.horizon, self.state_dim)

        return x

    def loss(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        x = self.expand_x(x)
        if cond is not None:
            x0 = apply_condition(x0, cond)
        x0 = self.expand_x(x0)
        x0, x = OT_sample(x0, x)
        xt = self.sample_conditional_pt(x0, x, t, sigma=0.01)
        ut = self.compute_conditional_vector_field(x0, x)
        vt = self.model(xt, t)
        loss = F.mse_loss(vt, ut)
        return loss