"""Arm-state scramble collector — kills the distribution shift that sank the
laundry diffusion round.

--goal flat  : each episode CRUMPLES the garment into the bucket, arm-PICKS
               it onto the table (the real eval-time pipeline), then applies
               armless scramble folds from that dropped state, recording
               inverse (grasp, place) pairs. States now match eval exactly.
--goal folded: canonical two-fold state TELEPORTED to the arm-drop position/
               orientation distribution (offset + yaw sampled from measured
               drop stats) + jitter, then scramble folds. (A real arm cannot
               place a folded garment without unfolding it, so the drop
               distribution is matched instead.)

Episodes that explode (any 16x16 heightmap cell > 0.30 m) are dropped
mid-episode by the inline filter.

Usage: python collect_armstate.py --garment tshirt4.obj --goal flat \
    --episodes 150 --shard 0 --out data_armstate
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--garment", type=str, default="tshirt4.obj")
ap.add_argument("--goal", type=str, default="flat", choices=["flat", "folded"])
ap.add_argument("--episodes", type=int, default=150)
ap.add_argument("--max_folds", type=int, default=2)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--out", type=str, default="data_armstate")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import scene as base
base.PARTICLE_SIZE = 0.008 if "pants" in args.garment else 0.0072
base.SUBSTEPS = 15
from scene_laundry import build_laundry_scene, crumple_into_bucket
from primitives_laundry import LaundryDuo
from scene import settle, HOME_L, HOME_R, HOME_I, TABLE_H
from metrics import particles, coverage
from garment_folds_lib import reflect_fold, canonical_seq_for

HM = 16
WS = 0.35
EXPLODED_MAX = 0.30 / 0.08
EHM = 32                 # env-style obs resolution (for behavior cloning)
EWS = 0.38

scene, left, right, ironer, cloths, cam = build_laundry_scene(
    [args.garment], seed=2600 + args.shard)
scene.build()
for e, h in [(left, HOME_L), (right, HOME_R), (ironer, HOME_I)]:
    e.set_dofs_position(h)
    e.control_dofs_position(h)
settle(scene, 150)
cloth = cloths[0]
flat = particles(cloth).copy()
duo = LaundryDuo(scene, left, right, cloth, flat_pos=flat)
rng = np.random.default_rng(6000 + args.shard)

folded_goal = flat.copy()
for pt, n2 in canonical_seq_for("tshirt.obj", flat):   # same 2-fold recipe
    d = (folded_goal[:, :2] - pt) @ n2
    folded_goal = reflect_fold(folded_goal, pt, n2, d > 0)


def obs_of(pos):
    hm = np.zeros((HM, HM), dtype=np.float32)
    ij = np.clip(((pos[:, :2] + WS) / (2 * WS) * HM).astype(int), 0, HM - 1)
    for (i, j), z in zip(ij, pos[:, 2] - TABLE_H):
        if z > hm[i, j]:
            hm[i, j] = z
    return np.concatenate([hm.flatten() / 0.08,
                           [coverage(pos)]]).astype(np.float32)


def obs_env(pos):
    """LaundryEnv-compatible obs (32x32 hm + 6 scalars) for BC training."""
    hm = np.zeros((EHM, EHM), dtype=np.float32)
    ij = np.clip(((pos[:, :2] + EWS) / (2 * EWS) * EHM).astype(int),
                 0, EHM - 1)
    for (i, j), z in zip(ij, pos[:, 2] - TABLE_H):
        if z > hm[i, j]:
            hm[i, j] = z
    cov = coverage(pos)
    ij2 = np.floor(pos[:, :2] / 0.02).astype(int)
    area = len({(int(a), int(b)) for a, b in ij2}) * 0.02 * 0.02
    flat_ij = np.floor(flat[:, :2] / 0.02).astype(int)
    flat_area = len({(int(a), int(b)) for a, b in flat_ij}) * 0.02 * 0.02
    ratio = area / max(flat_area, 1e-6)
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    rect = min(1.0, area / max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-6))
    phase = 1.0 if cov > 0.70 else 0.0
    return np.concatenate([hm.flatten() / 0.06,
                           [cov, ratio, rect, phase, 0.0, 0.0]]
                          ).astype(np.float32)


def act8_of(pos, g1xy, g2xy, p1xy, p2xy):
    """Inverse of LaundryEnv._to_xy: xy -> action in [-1,1] over cloth bbox."""
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    span = np.maximum(hi - lo, 1e-6)

    def inv(xy):
        return np.clip(2 * (np.asarray(xy) - lo) / span - 1, -1, 1)
    if g1xy[0] > g2xy[0]:
        g1xy, g2xy, p1xy, p2xy = g2xy, g1xy, p2xy, p1xy
    return np.concatenate([inv(g1xy), inv(g2xy), inv(p1xy), inv(p2xy)]
                          ).astype(np.float32)


def cell_of(xy):
    ij = np.clip(((np.asarray(xy) + WS) / (2 * WS) * HM).astype(int),
                 0, HM - 1)
    return int(ij[0] * HM + ij[1])


def scramble_once(pos):
    center = pos[:, :2].mean(0)
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
    new = reflect_fold(pos, pt, n2, side)
    t = np.array([-n2[1], n2[0]])
    flap = np.where(side)[0]
    dd = np.abs(d[flap])
    far = flap[dd > dd.max() * 0.6]
    if len(far) == 0:
        far = flap[np.argsort(dd)[-4:]]
    tt = (new[far, :2] - pt) @ t
    c1, c2 = int(far[np.argmin(tt)]), int(far[np.argmax(tt)])
    return new, c1, c2, pre[c1, :2].copy(), pre[c2, :2].copy()


def reset_arm_state():
    """flat goal: real crumple + arm pick. Returns True if usable."""
    try:
        duo.release_all()
        duo.home(seconds=0.5)
        cloth.set_particles_pos(flat + [0, 0, 0.02])
        cloth.set_particles_vel(np.zeros_like(flat))
        settle(scene, 40)
        crumple_into_bucket(scene, cloth, rng, folds=int(rng.integers(2, 4)),
                            settle_steps=150)
        ok = duo.pick_from_bucket()
    except Exception:
        return False
    return bool(ok)


def reset_folded_dropstyle():
    """folded goal: canonical fold placed with drop-like offset/yaw+jitter."""
    try:
        duo.release_all()
        duo.home(seconds=0.5)
    except Exception:
        return False
    p = folded_goal.copy()
    th = rng.uniform(-np.pi, np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = p[:, :2].mean(0)
    p[:, :2] = (p[:, :2] - c) @ R.T + c
    p[:, :2] += rng.normal([-0.05, 0.0], [0.10, 0.10])   # drop-position stats
    p += rng.normal(0, 0.004, p.shape)                   # drop rumple
    p[:, 2] += 0.02
    cloth.set_particles_pos(p)
    cloth.set_particles_vel(np.zeros_like(p))
    duo.settle_damped(120)
    q = particles(cloth)
    return bool(np.isfinite(q).all()
                and q[:, 2].max() - TABLE_H < 0.30)


X, G1, G2, P1, P2, OBS32, ACT8 = [], [], [], [], [], [], []
tag = (f"{os.path.splitext(args.garment)[0]}_{args.goal}_{args.shard}")
for ep in range(args.episodes):
    ok = reset_arm_state() if args.goal == "flat" else reset_folded_dropstyle()
    if not ok:
        continue
    for f in range(int(rng.integers(1, args.max_folds + 1))):
        r = scramble_once(particles(cloth))
        if r is None:
            break
        new, c1, c2, pre1, pre2 = r
        cloth.set_particles_pos(new)
        cloth.set_particles_vel(np.zeros_like(new))
        duo.settle_damped(160)
        pos2 = particles(cloth)
        ob = obs_of(pos2)
        if ob[:HM * HM].max() > EXPLODED_MAX:
            break
        X.append(ob)
        G1.append(cell_of(pos2[c1, :2]))
        G2.append(cell_of(pos2[c2, :2]))
        P1.append(cell_of(pre1))
        P2.append(cell_of(pre2))
        OBS32.append(obs_env(pos2))
        ACT8.append(act8_of(pos2, pos2[c1, :2], pos2[c2, :2], pre1, pre2))
    if (ep + 1) % 10 == 0:
        print(f"{tag}: {ep+1}/{args.episodes} eps, {len(X)} pairs", flush=True)
        np.savez(os.path.join(args.out, f"{tag}.npz"),
                 X=np.array(X), G1=np.array(G1), G2=np.array(G2),
                 P1=np.array(P1), P2=np.array(P2),
                 OBS32=np.array(OBS32), ACT8=np.array(ACT8))
np.savez(os.path.join(args.out, f"{tag}.npz"),
         X=np.array(X), G1=np.array(G1), G2=np.array(G2),
         P1=np.array(P1), P2=np.array(P2),
         OBS32=np.array(OBS32), ACT8=np.array(ACT8))
print(f"{tag} DONE: {len(X)} pairs", flush=True)
