"""Compact SAC for the fling env (low-dim obs/action, off-policy for sample
efficiency — cloth primitives are expensive)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(inp, out, hidden=(256, 256)):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd.split(self.act_dim, -1)
        log_std = log_std.clamp(-5, 2)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre = mu + eps * std
        a = torch.tanh(pre)
        logp = (-0.5 * ((eps) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1)
        logp -= (2 * (np.log(2) - pre - F.softplus(-2 * pre))).sum(-1)
        return a, logp

    def det(self, obs):
        mu, _ = self(obs)
        return torch.tanh(mu)


class SAC:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.95, tau=0.005):
        self.actor = Actor(obs_dim, act_dim).to(DEV)
        self.q1 = mlp(obs_dim + act_dim, 1).to(DEV)
        self.q2 = mlp(obs_dim + act_dim, 1).to(DEV)
        self.q1t = mlp(obs_dim + act_dim, 1).to(DEV)
        self.q2t = mlp(obs_dim + act_dim, 1).to(DEV)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.log_alpha = torch.tensor(np.log(0.2), device=DEV, requires_grad=True)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim
        self.gamma, self.tau = gamma, tau

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=DEV).unsqueeze(0)
            a = self.actor.det(o) if deterministic else self.actor.sample(o)[0]
            return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        o, a, r, o2, d = (torch.as_tensor(x, dtype=torch.float32, device=DEV) for x in batch)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q_t = torch.min(self.q1t(torch.cat([o2, a2], -1)),
                            self.q2t(torch.cat([o2, a2], -1))).squeeze(-1)
            target = r + self.gamma * (1 - d) * (q_t - alpha * logp2)
        q1 = self.q1(torch.cat([o, a], -1)).squeeze(-1)
        q2 = self.q2(torch.cat([o, a], -1)).squeeze(-1)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        an, logp = self.actor.sample(o)
        q_pi = torch.min(self.q1(torch.cat([o, an], -1)),
                         self.q2(torch.cat([o, an], -1))).squeeze(-1)
        pi_loss = (self.log_alpha.exp().detach() * logp - q_pi).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        a_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return float(q_loss), float(pi_loss)


class Replay:
    def __init__(self, obs_dim, act_dim, cap=100_000):
        self.o = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros((cap, act_dim), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, o, a, r, o2, d):
        j = self.i
        self.o[j], self.a[j], self.r[j], self.o2[j], self.d[j] = o, a, r, o2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, bs)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]
