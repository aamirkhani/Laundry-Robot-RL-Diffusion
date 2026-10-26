"""Scramble-inversion dataset collection (per Khani 2026, Zenodo 21970238 &
22063663): the forward "diffusion" process applies single random reflection
folds to a FLAT towel; the training pair is (observation after the fold
settles, the inverse un-fold action = grasp the folded flap's two corners).

No arm execution needed — pure cloth-state scrambling, so data is ~50x cheaper
than RL environment interaction.

Usage: python collect_scramble.py --episodes 400 --shard 0 --out data_scramble
(one Genesis scene per process; run several shards in parallel)
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=400)
ap.add_argument("--max_folds", type=int, default=3)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--out", type=str, default="data_scramble")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from scene import build_scene, settle, hold_home, HOME_L, HOME_R
from metrics import particles, coverage

HM = 16          # heightmap resolution (obs = HM*HM + 1 dims)
WS = 0.35        # workspace half-extent for the heightmap window

scene, left, right, cloth, cam = build_scene(n_grid=31, seed=900 + args.shard)
scene.build()
left.set_dofs_position(HOME_L)
right.set_dofs_position(HOME_R)
hold_home(left, right)
settle(scene, 100)
flat = particles(cloth).copy()
rng = np.random.default_rng(9000 + args.shard)


def damp():
    pos = cloth.get_particles_pos().cpu().numpy()
    vel = cloth.get_particles_vel().cpu().numpy()
    if pos.ndim == 3:
        pos, vel = pos[0], vel[0]
    m = pos[:, 2] < 0.025
    vel[m, 0] = 0.0
    vel[m, 1] = 0.0
    s = np.linalg.norm(vel, axis=1)
    f = s > 3.0
    if f.any():
        vel[f] *= (3.0 / s[f])[:, None]
    cloth.set_particles_vel(vel)


def settle_damped(steps=160):
    for _ in range(steps // 10):
        settle(scene, 10)
        damp()


def obs_of(pos):
    hm = np.zeros((HM, HM), dtype=np.float32)
    ij = np.clip(((pos[:, :2] + WS) / (2 * WS) * HM).astype(int), 0, HM - 1)
    for (i, j), z in zip(ij, pos[:, 2]):
        hm[i, j] = max(hm[i, j], z)
    return np.concatenate([hm.flatten() / 0.08, [coverage(pos)]]).astype(np.float32)


def cell_of(xy):
    ij = np.clip(((np.asarray(xy) + WS) / (2 * WS) * HM).astype(int), 0, HM - 1)
    return int(ij[0] * HM + ij[1])


def apply_one_fold(pos):
    """One reflection fold (the forward scramble move). Returns the flap's two
    corner particle indices — their post-fold positions are the inverse-action
    grasp points."""
    center = pos[:, :2].mean(axis=0)
    for _ in range(20):
        pt = center + rng.uniform(-0.06, 0.06, 2)
        ang = rng.uniform(0, np.pi)
        n2 = np.array([np.cos(ang), np.sin(ang)])
        d = (pos[:, :2] - pt) @ n2
        side = d > 0
        if side.sum() > pos.shape[0] * 0.15 and (~side).sum() > pos.shape[0] * 0.15:
            break
    else:
        return None
    if side.sum() > (~side).sum():
        side = ~side
        d = -d
    new = pos.copy()
    new[side, :2] -= 2 * d[side, None] * n2
    new[side, 2] = new[side, 2].max() + new[~side, 2].max() + 0.014
    # flap corners: among the flap's farthest-from-crease particles, the two
    # tangential extremes (the reflected images of the flap's outer corners)
    t = np.array([-n2[1], n2[0]])
    flap_idx = np.where(side)[0]
    dd = d[flap_idx]                        # pre-fold distance from the crease (>0)
    far = flap_idx[dd > dd.max() * 0.6]
    if len(far) == 0:
        far = flap_idx[np.argsort(dd)[-4:]]
    tt = (new[far, :2] - pt) @ t
    c1 = far[np.argmin(tt)]
    c2 = far[np.argmax(tt)]
    return new, int(c1), int(c2)


X, Y1, Y2, XY = [], [], [], []
for ep in range(args.episodes):
    cloth.set_particles_pos(flat + [0, 0, 0.02])
    cloth.set_particles_vel(np.zeros_like(flat))
    settle_damped(60)
    k = int(rng.integers(1, args.max_folds + 1))
    for f in range(k):
        pos = particles(cloth)
        r = apply_one_fold(pos)
        if r is None:
            break
        new, c1, c2 = r
        cloth.set_particles_pos(new)
        cloth.set_particles_vel(np.zeros_like(new))
        settle_damped(160)
        # a training pair at EVERY noise level: obs after this fold settles ->
        # inverse of this (last) fold
        pos2 = particles(cloth)
        X.append(obs_of(pos2))
        Y1.append(cell_of(pos2[c1, :2]))
        Y2.append(cell_of(pos2[c2, :2]))
        XY.append(np.concatenate([pos2[c1, :2], pos2[c2, :2]]))
    if (ep + 1) % 25 == 0:
        print(f"shard {args.shard}: {ep+1}/{args.episodes} eps, {len(X)} samples",
              flush=True)
        np.savez(os.path.join(args.out, f"shard_{args.shard}.npz"),
                 X=np.array(X), Y1=np.array(Y1), Y2=np.array(Y2), XY=np.array(XY))

np.savez(os.path.join(args.out, f"shard_{args.shard}.npz"),
         X=np.array(X), Y1=np.array(Y1), Y2=np.array(Y2), XY=np.array(XY))
print(f"shard {args.shard} DONE: {len(X)} samples", flush=True)
