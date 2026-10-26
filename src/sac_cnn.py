"""SAC with a CNN encoder over the heightmap observation (laundry v2).

Obs layout: [HM*HM heightmap, k scalars]. The conv trunk embeds the map;
scalars are concatenated to the embedding. Bigger than the MLP used in v1
per the "more complex model" directive.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, hm, n_scalar, feat=256):
        super().__init__()
        self.hm = hm
        self.n_scalar = n_scalar
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # hm/2
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # hm/4
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(),
        )
        n_flat = 128 * (hm // 4) * (hm // 4)
        self.fc = nn.Sequential(nn.Linear(n_flat + n_scalar, feat), nn.ReLU(),
                                nn.Linear(feat, feat), nn.ReLU())

    def forward(self, x):
        m = x[:, : self.hm * self.hm].reshape(-1, 1, self.hm, self.hm)
        s = x[:, self.hm * self.hm:]
        h = self.conv(m).flatten(1)
        return self.fc(torch.cat([h, s], 1))


class Actor(nn.Module):
    def __init__(self, hm, n_scalar, act_dim, feat=256):
        super().__init__()
        self.enc = Encoder(hm, n_scalar, feat)
        self.mu = nn.Linear(feat, act_dim)
        self.log_std = nn.Linear(feat, act_dim)

    def forward(self, obs):
        h = self.enc(obs)
        return self.mu(h), self.log_std(h).clamp(-5, 2)

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        pre = mu + std * torch.randn_like(mu)
        a = torch.tanh(pre)
        logp = (-0.5 * ((pre - mu) / std) ** 2 - log_std
                - 0.5 * np.log(2 * np.pi)).sum(1)
        logp = logp - torch.log(1 - a ** 2 + 1e-6).sum(1)
        return a, logp

    def det(self, obs):
        return torch.tanh(self(obs)[0])


class Critic(nn.Module):
    def __init__(self, hm, n_scalar, act_dim, feat=256):
        super().__init__()
        self.enc = Encoder(hm, n_scalar, feat)
        self.q = nn.Sequential(nn.Linear(feat + act_dim, 256), nn.ReLU(),
                               nn.Linear(256, 256), nn.ReLU(),
                               nn.Linear(256, 1))

    def forward(self, obs, act):
        return self.q(torch.cat([self.enc(obs), act], 1))


class SACCNN:
    def __init__(self, hm, n_scalar, act_dim, lr=3e-4, gamma=0.96, tau=0.005):
        self.actor = Actor(hm, n_scalar, act_dim).to(DEV)
        self.q1 = Critic(hm, n_scalar, act_dim).to(DEV)
        self.q2 = Critic(hm, n_scalar, act_dim).to(DEV)
        import copy
        self.q1t = copy.deepcopy(self.q1)
        self.q2t = copy.deepcopy(self.q2)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.log_alpha = torch.tensor(np.log(0.2), device=DEV,
                                      requires_grad=True)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim
        self.gamma, self.tau = gamma, tau

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32,
                                device=DEV).unsqueeze(0)
            a = self.actor.det(o) if deterministic else self.actor.sample(o)[0]
            return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        o, a, r, o2, d = [torch.as_tensor(x, dtype=torch.float32, device=DEV)
                          for x in batch]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            qt = torch.min(self.q1t(o2, a2), self.q2t(o2, a2)).squeeze(1)
            y = r + self.gamma * (1 - d) * (qt - alpha * logp2)
        q_loss = F.mse_loss(self.q1(o, a).squeeze(1), y) + \
            F.mse_loss(self.q2(o, a).squeeze(1), y)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        an, logp = self.actor.sample(o)
        q = torch.min(self.q1(o, an), self.q2(o, an)).squeeze(1)
        pi_loss = (alpha * logp - q).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        a_loss = -(self.log_alpha.exp() *
                   (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()

        with torch.no_grad():
            for src, dst in [(self.q1, self.q1t), (self.q2, self.q2t)]:
                for ps, pd in zip(src.parameters(), dst.parameters()):
                    pd.mul_(1 - self.tau).add_(self.tau * ps)
        return float(q_loss.detach()), float(pi_loss.detach())


class Replay:
    def __init__(self, obs_dim, act_dim, cap=60_000):
        self.o = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros((cap, act_dim), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, o, a, r, o2, d):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, bs)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]
