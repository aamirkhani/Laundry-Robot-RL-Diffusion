"""Behavior-clone the scramble-inverse policy into the CNN actor.

Data: (OBS32, ACT8) pairs recorded by collect_armstate.py — the env-style
observation and the known-correct inverse action for each scrambled state.
The actor's tanh(mu) regresses the action (MSE in pre-tanh space via atanh
targets, clipped for stability). The result warm-starts SAC fine-tuning.

Usage: python bc_train.py --data data_armstate --garment tshirt4 \
    --out bc_actor_tshirt4.pt --epochs 200
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=str, default="data_armstate")
ap.add_argument("--garment", type=str, default="tshirt4")
ap.add_argument("--out", type=str, default="bc_actor.pt")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--torch_seed", type=int, default=0)
args = ap.parse_args()

from sac_cnn import Actor

O, A = [], []
for f in sorted(glob.glob(f"{args.data}/{args.garment}*_*.npz")):
    d = np.load(f)
    if "OBS32" in d and d["OBS32"].size:
        O.append(d["OBS32"])
        A.append(d["ACT8"])
O = np.concatenate(O)
A = np.concatenate(A)
print(f"BC data: {len(O)} pairs from {args.garment}")

torch.manual_seed(args.torch_seed)
actor = Actor(32, 6, 8).cuda()
opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
Ot = torch.as_tensor(O, dtype=torch.float32, device="cuda")
At = torch.as_tensor(np.clip(A, -0.999, 0.999), dtype=torch.float32,
                     device="cuda")
pre_target = torch.atanh(At)

n = len(O)
idx = np.random.default_rng(0).permutation(n)
val, trn = idx[: n // 10], idx[n // 10:]

for ep in range(args.epochs):
    actor.train()
    perm = np.random.permutation(trn)
    tot = 0.0
    for k in range(0, len(perm), 256):
        b = perm[k:k + 256]
        mu, _ = actor(Ot[b])
        loss = F.mse_loss(mu, pre_target[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    if ep % 40 == 39 or ep == 0:
        actor.eval()
        with torch.no_grad():
            mu, _ = actor(Ot[val])
            err = (torch.tanh(mu) - At[val]).abs().mean()
        print(f"epoch {ep+1}: train {tot/len(perm):.4f} "
              f"| val mean |a-a*| {float(err):.3f}", flush=True)

torch.save({"actor": actor.state_dict()}, args.out)
print(f"saved {args.out}")
