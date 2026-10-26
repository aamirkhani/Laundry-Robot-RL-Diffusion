"""Scramble-inversion data v2 — garments + both goals.

--goal flat   : forward scramble from the FLAT garment (untangling data)
--goal folded : forward scramble from the canonically FOLDED garment
                (folding data — reverse rollout folds the garment)

Each pair records the inverse action: grasp cells (flap corners, current pos)
AND place cells (where those corners came from — their pre-fold positions).

Usage: python collect_scramble2.py --garment tshirt.obj --goal folded --shard 0
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--garment", type=str, default="")     # '' = towel
ap.add_argument("--goal", type=str, default="flat", choices=["flat", "folded"])
ap.add_argument("--episodes", type=int, default=250)
ap.add_argument("--max_folds", type=int, default=3)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--out", type=str, default="data_scramble2")
ap.add_argument("--psize", type=float, default=0.0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import scene as _base
if __name__ == "__main__" or True:
    pass
from scene import build_scene, settle, hold_home, HOME_L, HOME_R, TABLE_H
if args.psize > 0:
    _base.PARTICLE_SIZE = args.psize
    _base.SUBSTEPS = 15
from metrics import particles, coverage

HM = 16
WS = 0.35

scene, left, right, cloth, cam = build_scene(
    garment=args.garment or None, seed=1700 + args.shard)
scene.build()
left.set_dofs_position(HOME_L)
right.set_dofs_position(HOME_R)
hold_home(left, right)
settle(scene, 100)
flat = particles(cloth).copy()
rng = np.random.default_rng(4000 + args.shard)


def damp():
    pos = cloth.get_particles_pos().cpu().numpy()
    vel = cloth.get_particles_vel().cpu().numpy()
    if pos.ndim == 3:
        pos, vel = pos[0], vel[0]
    m = pos[:, 2] < TABLE_H + 0.025
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
    for (i, j), z in zip(ij, pos[:, 2] - TABLE_H):
        hm[i, j] = max(hm[i, j], z)
    return np.concatenate([hm.flatten() / 0.08, [coverage(pos)]]).astype(np.float32)


def cell_of(xy):
    ij = np.clip(((np.asarray(xy) + WS) / (2 * WS) * HM).astype(int), 0, HM - 1)
    return int(ij[0] * HM + ij[1])


def reflect(pos, pt, n2, side):
    from garment_folds_lib import reflect_fold
    return reflect_fold(pos, pt, n2, side)


def canonical_folded(flat_pos):
    """Ideal folded configuration, synthetically (no arms) — shared defs."""
    from garment_folds_lib import canonical_seq_for
    p = flat_pos.copy()
    for pt, n2 in canonical_seq_for(args.garment or "towel", flat_pos):
        d = (p[:, :2] - pt) @ n2
        p = reflect(p, pt, n2, d > 0)
    return p


def apply_one_scramble_fold(pos):
    """Random reflection fold; returns (new_pos, flap corner idx pair, their
    pre-fold positions)."""
    center = pos[:, :2].mean(axis=0)
    for _ in range(20):
        pt = center + rng.uniform(-0.06, 0.06, 2)
        ang = rng.uniform(0, np.pi)
        n2 = np.array([np.cos(ang), np.sin(ang)])
        d = (pos[:, :2] - pt) @ n2
        side = d > 0
        if side.sum() > len(pos) * 0.15 and (~side).sum() > len(pos) * 0.15:
            break
    else:
        return None
    if side.sum() > (~side).sum():
        side = ~side
        d = -d
    pre = pos.copy()
    new = reflect(pos, pt, n2, side)
    t = np.array([-n2[1], n2[0]])
    flap_idx = np.where(side)[0]
    dd = np.abs(d[flap_idx])
    far = flap_idx[dd > dd.max() * 0.6]
    if len(far) == 0:
        far = flap_idx[np.argsort(dd)[-4:]]
    tt = (new[far, :2] - pt) @ t
    c1, c2 = int(far[np.argmin(tt)]), int(far[np.argmax(tt)])
    return new, c1, c2, pre[c1, :2].copy(), pre[c2, :2].copy()


goal = flat if args.goal == "flat" else canonical_folded(flat)

X, G1, G2, P1, P2 = [], [], [], [], []
for ep in range(args.episodes):
    cloth.set_particles_pos(goal + [0, 0, 0.02])
    cloth.set_particles_vel(np.zeros_like(goal))
    settle_damped(80)
    base = particles(cloth)
    k = int(rng.integers(1, args.max_folds + 1))
    cur = base
    for f in range(k):
        r = apply_one_scramble_fold(particles(cloth))
        if r is None:
            break
        new, c1, c2, pre1, pre2 = r
        cloth.set_particles_pos(new)
        cloth.set_particles_vel(np.zeros_like(new))
        settle_damped(160)
        pos2 = particles(cloth)
        ob = obs_of(pos2)
        EXPLODED_MAX = 0.30 / 0.08     # any hm cell above 0.30m = solver blew up
        if ob[:HM * HM].max() > EXPLODED_MAX:
            break                       # drop sample AND stop this episode
        X.append(ob)
        G1.append(cell_of(pos2[c1, :2]))
        G2.append(cell_of(pos2[c2, :2]))
        P1.append(cell_of(pre1))
        P2.append(cell_of(pre2))
    if (ep + 1) % 25 == 0:
        tag = f"{os.path.splitext(args.garment or 'towel')[0]}_{args.goal}_{args.shard}"
        print(f"{tag}: {ep+1}/{args.episodes} eps, {len(X)} pairs", flush=True)
        np.savez(os.path.join(args.out, f"{tag}.npz"),
                 X=np.array(X), G1=np.array(G1), G2=np.array(G2),
                 P1=np.array(P1), P2=np.array(P2))

tag = f"{os.path.splitext(args.garment or 'towel')[0]}_{args.goal}_{args.shard}"
np.savez(os.path.join(args.out, f"{tag}.npz"),
         X=np.array(X), G1=np.array(G1), G2=np.array(G2),
         P1=np.array(P1), P2=np.array(P2))
print(f"{tag} DONE: {len(X)} pairs", flush=True)
