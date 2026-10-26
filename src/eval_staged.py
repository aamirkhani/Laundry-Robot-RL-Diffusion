"""End-to-end staged evaluation: tangled garment -> untangle -> two-fold.

Modes (each runs on the same tangle seeds, same metric):
  diffusion : untangle-denoiser flings until coverage OK, then fold-denoiser
              reverse rollout (the scramble-inversion method)
  sac       : SAC policy from train_staged.py checkpoint
  oracle    : scripted upper bound — oracle un-folds + canonical folds

Metric: final goal_err (Procrustes distance to canonical two-fold state) and
final coverage; success = goal_err < 0.06.

Usage:
  python eval_staged.py --garment tshirt.obj --episodes 6 \
      --modes diffusion,sac --sac sac_staged.pt \
      --untangle untangle_tshirt.pt --fold fold_tshirt.pt [--video out.mp4]
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--garment", type=str, default="tshirt.obj")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--modes", type=str, default="diffusion")
ap.add_argument("--untangle", type=str, default="untangle_denoiser.pt")
ap.add_argument("--fold", type=str, default="fold_denoiser.pt")
ap.add_argument("--sac", type=str, default="sac_staged.pt")
ap.add_argument("--max_flings", type=int, default=4)
ap.add_argument("--max_folds", type=int, default=4)
ap.add_argument("--video", type=str, default="")
ap.add_argument("--out", type=str, default="eval_staged.json")
args = ap.parse_args()

import torch

from scene import (build_scene, settle, hold_home, spawn_tangled,
                   HOME_L, HOME_R, TABLE_H)
from primitives import Duo
from metrics import particles, coverage
from train_denoiser2 import Denoiser4, HM
from garment_folds_lib import canonical_seq_for, reflect_fold

WS = 0.35
COV_OK = 0.72
GOAL_OK = 0.06

scene, left, right, cloth, cam = build_scene(garment=args.garment, seed=777)
scene.build()
left.set_dofs_position(HOME_L)
right.set_dofs_position(HOME_R)
hold_home(left, right)
settle(scene, 100)
flat = particles(cloth).copy()
duo = Duo(scene, left, right, cloth, cam=cam, record=bool(args.video),
          flat_pos=flat)

goalp = flat.copy()
for pt, n2 in canonical_seq_for(args.garment, flat):
    d = (goalp[:, :2] - pt) @ n2
    goalp = reflect_fold(goalp, pt, n2, d > 0)


def goal_err(pos):
    a = pos[:, :2] - pos[:, :2].mean(0)
    b = goalp[:, :2] - goalp[:, :2].mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return float(np.linalg.norm(a @ R - b, axis=1).mean())


def obs16(pos):
    hm = np.zeros((HM, HM), dtype=np.float32)
    ij = np.clip(((pos[:, :2] + WS) / (2 * WS) * HM).astype(int), 0, HM - 1)
    for (i, j), z in zip(ij, pos[:, 2] - TABLE_H):
        hm[i, j] = max(hm[i, j], z)
    return np.concatenate([hm.flatten() / 0.08,
                           [coverage(pos)]]).astype(np.float32)


def cell_particle(cell, pos):
    ci, cj = cell // HM, cell % HM
    lo = np.array([-WS + ci * 2 * WS / HM, -WS + cj * 2 * WS / HM])
    hi = lo + 2 * WS / HM
    inside = np.where((pos[:, 0] >= lo[0]) & (pos[:, 0] < hi[0]) &
                      (pos[:, 1] >= lo[1]) & (pos[:, 1] < hi[1]))[0]
    if len(inside):
        return pos[inside[np.argmax(pos[inside, 2])], :2]
    return lo + WS / HM


def cell_center(cell):
    ci, cj = cell // HM, cell % HM
    return np.array([-WS + (ci + 0.5) * 2 * WS / HM,
                     -WS + (cj + 0.5) * 2 * WS / HM])


def load_denoiser(path):
    net = Denoiser4().cuda()
    ck = torch.load(path, map_location="cuda")
    net.load_state_dict(ck["net"] if "net" in ck else ck)
    net.eval()
    return net


unet = load_denoiser(args.untangle) if os.path.exists(args.untangle) else None
fnet = load_denoiser(args.fold) if os.path.exists(args.fold) else None

sac_actor = None
if os.path.exists(args.sac) and "sac" in args.modes:
    from sac import SAC
    agent = SAC(obs_dim=HM * HM + 3, act_dim=8)
    ck = torch.load(args.sac, map_location="cuda")
    agent.actor.load_state_dict(ck["actor"])
    agent.actor.eval()
    sac_actor = agent.actor


def sac_obs(pos):
    cov = coverage(pos)
    ge = goal_err(pos)
    phase = 1.0 if cov > COV_OK else 0.0
    return np.concatenate([obs16(pos)[:HM * HM],
                           [cov, ge, phase]]).astype(np.float32)


def to_xy(a2, pos):
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    t = (np.asarray(a2) + 1) / 2
    return lo + t * (hi - lo)


def fling_particles(i, j):
    p = particles(cloth)
    if p[i, 0] > p[j, 0]:
        i, j = j, i
    duo.fling(None, None, particles_lr=(i, j))


def run_diffusion():
    # phase 1: untangle via reverse rollout of the untangle-denoiser
    for _ in range(args.max_flings):
        pos = particles(cloth)
        if coverage(pos) > COV_OK:
            break
        with torch.no_grad():
            heads = unet(torch.as_tensor(obs16(pos)).cuda().unsqueeze(0))
        g1 = cell_particle(int(heads[0][0].argmax()), pos)
        g2 = cell_particle(int(heads[1][0].argmax()), pos)
        i = int(np.argmin(((pos[:, :2] - g1) ** 2).sum(1)))
        j = int(np.argmin(((pos[:, :2] - g2) ** 2).sum(1)))
        if i == j:
            j = int(np.argmax(((pos[:, :2] - pos[i, :2]) ** 2).sum(1)))
        try:
            fling_particles(i, j)
        except Exception as e:
            print(f"    fling crashed: {e}")
            break
    # phase 2: fold via reverse rollout of the fold-denoiser
    prev = goal_err(particles(cloth))
    for _ in range(args.max_folds):
        pos = particles(cloth)
        with torch.no_grad():
            heads = fnet(torch.as_tensor(obs16(pos)).cuda().unsqueeze(0))
        c1, c2 = int(heads[0][0].argmax()), int(heads[1][0].argmax())

        def far_place(logits, gcell):
            gi, gj = gcell // HM, gcell % HM
            for c in torch.topk(logits, 8).indices.tolist():
                if abs(c // HM - gi) + abs(c % HM - gj) >= 3:
                    return c
            return int(logits.argmax())

        g1, g2 = cell_particle(c1, pos), cell_particle(c2, pos)
        p1 = cell_center(far_place(heads[2][0], c1))
        p2 = cell_center(far_place(heads[3][0], c2))
        if g1[0] > g2[0]:
            g1, g2, p1, p2 = g2, g1, p2, p1
        try:
            duo.fold([*g1], [*g2], [*p1], [*p2])
        except Exception as e:
            print(f"    fold crashed: {e}")
            break
        err = goal_err(particles(cloth))
        if err > prev - 0.004:
            break
        prev = err


def run_sac():
    pos = particles(cloth)
    for _ in range(8):
        o = torch.as_tensor(sac_obs(pos)).cuda().unsqueeze(0)
        with torch.no_grad():
            a = sac_actor.det(o).squeeze(0).cpu().numpy()
        pos = particles(cloth)
        cov = coverage(pos)
        try:
            if cov <= COV_OK:
                g1, g2 = to_xy(a[0:2], pos), to_xy(a[2:4], pos)
                i = int(np.argmin(((pos[:, :2] - g1) ** 2).sum(1)))
                j = int(np.argmin(((pos[:, :2] - g2) ** 2).sum(1)))
                fling_particles(i, j)
            else:
                g1, g2 = to_xy(a[0:2], pos), to_xy(a[2:4], pos)
                q1, q2 = to_xy(a[4:6], pos), to_xy(a[6:8], pos)
                if g1[0] > g2[0]:
                    g1, g2, q1, q2 = g2, g1, q2, q1
                duo.fold([*g1], [*g2], [*q1], [*q2])
        except Exception as e:
            print(f"    sac action crashed: {e}")
            break
        pos = particles(cloth)
        if goal_err(pos) < GOAL_OK and coverage(pos) > 0.4:
            break


def run_oracle(oracle):
    # un-fold using the known inverse (first grasp pair), then fling to flat,
    # then scripted canonical folds — the upper bound
    if oracle is not None:
        try:
            fling_particles(*oracle)
        except Exception:
            pass
    for _ in range(2):
        pos = particles(cloth)
        if coverage(pos) > COV_OK:
            break
        ext = pos[:, :2].max(0) - pos[:, :2].min(0)
        ax = int(np.argmax(ext))
        i = int(np.argmin(pos[:, ax]))
        j = int(np.argmax(pos[:, ax]))
        try:
            fling_particles(i, j)
        except Exception:
            break
    pos0 = particles(cloth)
    for pt, n2 in canonical_seq_for(args.garment, flat):
        pos = particles(cloth)
        d = (flat[:, :2] - pt) @ n2
        side = d > 0
        t = np.array([-n2[1], n2[0]])
        flap = np.where(side)[0]
        dd = d[flap]
        far = flap[dd > dd.max() * 0.6]
        tt = (flat[far, :2] - pt) @ t
        g1i, g2i = int(far[np.argmin(tt)]), int(far[np.argmax(tt)])
        m1 = flat[g1i, :2] - 2 * ((flat[g1i, :2] - pt) @ n2) * n2
        m2 = flat[g2i, :2] - 2 * ((flat[g2i, :2] - pt) @ n2) * n2
        q1i = int(np.argmin(((flat[:, :2] - m1) ** 2).sum(1)))
        q2i = int(np.argmin(((flat[:, :2] - m2) ** 2).sum(1)))
        g1, g2 = pos[g1i, :2], pos[g2i, :2]
        q1, q2 = pos[q1i, :2], pos[q2i, :2]
        if g1[0] > g2[0]:
            g1, g2, q1, q2 = g2, g1, q2, q1
        try:
            duo.fold([*g1], [*g2], [*q1], [*q2])
        except Exception as e:
            print(f"    oracle fold crashed: {e}")
            break


modes = [m for m in args.modes.split(",") if m]
if args.video:
    cam.start_recording()
results = {m: [] for m in modes}
for ep in range(args.episodes):
    for mode in modes:
        if mode == "diffusion" and (unet is None or fnet is None):
            continue
        if mode == "sac" and sac_actor is None:
            continue
        duo.release_all()
        duo.home(seconds=0.4)
        cloth.set_particles_pos(flat + [0, 0, 0.02])
        cloth.set_particles_vel(np.zeros_like(flat))
        duo.settle_damped(60)
        rng = np.random.default_rng(9000 + ep)   # same tangle per mode
        oracle = spawn_tangled(scene, cloth, rng, settle_steps=200,
                               return_oracle=True)
        c0 = coverage(particles(cloth))
        if mode == "diffusion":
            run_diffusion()
        elif mode == "sac":
            run_sac()
        else:
            run_oracle(oracle)
        pos = particles(cloth)
        ge, cov = goal_err(pos), coverage(pos)
        results[mode].append({"goal_err": ge, "coverage": cov,
                              "start_cov": float(c0),
                              "success": bool(ge < GOAL_OK and cov > 0.4)})
        print(f"ep {ep} {mode}: start_cov {c0:.2f} -> goal_err {ge:.3f} "
              f"cov {cov:.3f} success {ge < GOAL_OK and cov > 0.4}", flush=True)
if args.video:
    cam.stop_recording()
    mp4 = sorted(glob.glob("*_cam_0_*.mp4"), key=os.path.getmtime)[-1]
    os.rename(mp4, args.video)

summary = {}
for m, rs in results.items():
    if rs:
        summary[m] = {
            "mean_goal_err": float(np.mean([r["goal_err"] for r in rs])),
            "success_rate": float(np.mean([r["success"] for r in rs])),
            "mean_cov": float(np.mean([r["coverage"] for r in rs])),
            "n": len(rs),
        }
print(json.dumps(summary, indent=1))
with open(args.out, "w") as f:
    json.dump({**summary, "raw": results}, f, indent=1)
