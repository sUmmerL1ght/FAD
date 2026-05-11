import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
from torchcfm.optimal_transport import OTPlanSampler
import numpy as np

ot_sampler = OTPlanSampler(method='exact')


def OT_sample(x0, x1):
    return ot_sampler.sample_plan(x0, x1)


def apply_condition(seq, cond):
    for key, value in cond.items():
        seq[:, key] = value.clone()
    return seq


class FlowMatchingBase(nn.Module):
    def __init__(self, state_dim, model=None, horizon=100, device="cuda", hidden_dim=256):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_conditional_pt(self, x0, x1, t, sigma=0.01):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        return t * x1 + (1 - t) * x0 + sigma * torch.randn_like(x0)


class FlowMatching(FlowMatchingBase):
    """ODE-based flow matching using torchdiffeq (primary method used in paper)."""

    def forward(self, bs, num_steps=1, cond=None, reward=None):
        x_ini = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x_ini = apply_condition(x_ini, cond)
        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, cond=reward, time=torch.ones(bs).to(self.device) * t, use_dropout=False),
            x_ini,
            torch.linspace(0, 1, num_steps + 1).to(self.device),
            atol=1e-4, rtol=1e-4, method="euler",
        )
        return traj[-1]

    def loss(self, x, cond, reward):
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        if cond is not None:
            x0 = apply_condition(x0, cond)
        xt = self.sample_conditional_pt(x0, x, t)
        ut = self.compute_conditional_vector_field(x0, x)
        vt = self.model(xt, reward, t)
        return F.mse_loss(vt, ut)


def _euler_integrate(model, x, bs, num_steps, device, cond=None):
    dt = 1.0 / num_steps
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones(bs).to(device) * i * dt
            x = x + model(x, t) * dt
    return x


class FlowMatching_euler(FlowMatchingBase):
    def forward(self, bs, num_steps=100, cond=None):
        x = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x = apply_condition(x, cond)
        return _euler_integrate(self.model, x, bs, num_steps, self.device)

    def loss(self, x, cond=None):
        if cond is not None:
            x = apply_condition(x, cond)
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        xt = self.sample_conditional_pt(x0, x, t)
        ut = self.compute_conditional_vector_field(x0, x)
        return F.mse_loss(self.model(xt, t), ut)


class FlowMatching_unet_euler(FlowMatchingBase):
    def forward(self, bs, num_steps=100, cond=None):
        x = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x = apply_condition(x, cond)
        return _euler_integrate(self.model, x, bs, num_steps, self.device)

    def loss(self, x, cond=None):
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        if cond is not None:
            x0 = apply_condition(x0, cond)
        x0, x = OT_sample(x0, x)
        xt = self.sample_conditional_pt(x0, x, t)
        ut = self.compute_conditional_vector_field(x0, x)
        return F.mse_loss(self.model(xt, t), ut)


class FlowMatching_expand_mlp_euler(FlowMatchingBase):
    def _expand(self, x):
        return x.reshape(x.shape[0], -1)

    def forward(self, bs, num_steps=100, cond=None):
        x = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x = apply_condition(x, cond)
        x = self._expand(x)
        x = _euler_integrate(self.model, x, bs, num_steps, self.device)
        return x.reshape(bs, self.horizon, self.state_dim)

    def loss(self, x, cond=None):
        if cond is not None:
            x = apply_condition(x, cond)
        batch_size = x.shape[0]
        x = self._expand(x)
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        xt = self.sample_conditional_pt(x0, x, t)
        ut = self.compute_conditional_vector_field(x0, x)
        return F.mse_loss(self.model(xt, t), ut)


class FlowMatching_expand_mlp_euler_OT(FlowMatchingBase):
    def _expand(self, x):
        return x.reshape(x.shape[0], -1)

    def _unexpand(self, x):
        return x.reshape(x.shape[0], self.horizon, -1)

    def forward(self, bs, num_steps=100, cond=None):
        x = torch.randn(bs, self.horizon, self.state_dim).to(self.device)
        if cond is not None:
            x = apply_condition(x, cond)
        x = self._expand(x)
        dt = 1.0 / num_steps
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.ones(bs).to(self.device) * i * dt
                x = x + self.model(x, t) * dt
                x = self._unexpand(x)
                if cond is not None:
                    x = apply_condition(x, cond)
                x = self._expand(x)
        return x.reshape(bs, self.horizon, self.state_dim)

    def loss(self, x, cond=None):
        batch_size = x.shape[0]
        t = torch.rand(batch_size).type_as(x)
        x0 = torch.randn_like(x)
        x = self._expand(x)
        if cond is not None:
            x0 = apply_condition(x0, cond)
        x0 = self._expand(x0)
        x0, x = OT_sample(x0, x)
        xt = self.sample_conditional_pt(x0, x, t)
        ut = self.compute_conditional_vector_field(x0, x)
        return F.mse_loss(self.model(xt, t), ut)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.dim() == 1:
            return self.pe[:, x.long(), :].squeeze(1)
        return self.pe[:, :x.size(1), :]


class MLP_expand(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_emb_dim=16):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.SELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_emb_dim, w), nn.SELU(),
            nn.Linear(w, w), nn.SELU(),
            nn.Linear(w, w), nn.SELU(),
            nn.Linear(w, out_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_encoder(t.view(-1, 1))
        return self.net(torch.cat([x, t_emb], dim=1))


class TrajectoryFlowNetwork(nn.Module):
    def __init__(self, seq_len, feature_dim, hidden_dim=256):
        super().__init__()
        self.time_encoder = nn.Sequential(
            SinusoidalPositionalEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.sequence_encoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim // 2,
            bidirectional=True, batch_first=True
        )
        self.velocity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x, t):
        _, seq_len, _ = x.shape
        t_emb = self.time_encoder(t).squeeze(0).squeeze(1).unsqueeze(1).expand(-1, seq_len, -1)
        x_emb = self.feature_encoder(x)
        combined = x_emb + t_emb
        seq_features, _ = self.sequence_encoder(combined)
        return self.velocity_predictor(torch.cat([seq_features, combined], dim=-1))
