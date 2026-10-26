"""Validate the laundry-v2 policy: run full bucket episodes with the trained
CNN actor or a random policy, log per-garment outcomes, optional video.

Usage:
  python eval_laundry.py --actor runs_laundry/actor_final_laundry.pt \
      --episodes 3 --out eval_laundry_trained.json [--video laundry_rl.mp4]
  python eval_laundry.py --actor "" --episodes 3 --out eval_laundry_random.json
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--actor", type=str, default="runs_laundry/actor_final_laundry.pt")
ap.add_argument("--garment", type=str, default="tshirt4.obj")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--video", type=str, default="")
ap.add_argument("--out", type=str, default="eval_laundry.json")
ap.add_argument("--seed", type=int, default=500)
ap.add_argument("--mode", type=str, default="actor",
                choices=["actor", "diffusion"])
ap.add_argument("--uncrumple", type=str, default="uncrumple_tshirt4.pt")
ap.add_argument("--fold", type=str, default="fold_tshirt4.pt")
args = ap.parse_args()

import torch

from env_laundry import LaundryEnv, HM

env = LaundryEnv(seed=args.seed, garment=args.garment)

actor = None
dnets = None
if args.mode == "diffusion":
    from train_denoiser2 import Denoiser4, HM as DHM

    def _load(p):
        n = Denoiser4().cuda()
        ck = torch.load(p, map_location="cuda")
        n.load_state_dict(ck["net"] if "net" in ck else ck)
        n.eval()
        return n
    dnets = (_load(args.uncrumple), _load(args.fold))
    print(f"DIFFUSION policy: {args.uncrumple} + {args.fold}", flush=True)
elif args.actor and os.path.exists(args.actor):
    from sac_cnn import Actor
    actor = Actor(HM, 6, 8).cuda()
    ck = torch.load(args.actor, map_location="cuda")
    actor.load_state_dict(ck["actor"])
    actor.eval()
    print(f"loaded {args.actor}", flush=True)
else:
    print("RANDOM policy", flush=True)

DWS = 0.35


def diffusion_action(env, dnets):
    """Reverse-rollout step: denoiser grasp/place cells -> env action space.
    The env maps action [-1,1] over the ACTIVE cloth's bbox; invert that."""
    from metrics import particles, coverage
    from train_denoiser2 import HM as DHM
    unet, fnet = dnets
    pos = particles(env.duo.cloth)
    hm = np.zeros((DHM, DHM), dtype=np.float32)
    ij = np.clip(((pos[:, :2] + DWS) / (2 * DWS) * DHM).astype(int),
                 0, DHM - 1)
    from scene import TABLE_H
    for (i, j), z in zip(ij, pos[:, 2] - TABLE_H):
        if z > hm[i, j]:
            hm[i, j] = z
    o16 = np.concatenate([hm.flatten() / 0.08,
                          [coverage(pos)]]).astype(np.float32)
    net = unet if coverage(pos) <= 0.70 else fnet
    with torch.no_grad():
        heads = net(torch.as_tensor(o16).cuda().unsqueeze(0))

    def cell_xy(c, snap=True):
        ci, cj = c // DHM, c % DHM
        lo = np.array([-DWS + ci * 2 * DWS / DHM, -DWS + cj * 2 * DWS / DHM])
        hi = lo + 2 * DWS / DHM
        inside = np.where((pos[:, 0] >= lo[0]) & (pos[:, 0] < hi[0]) &
                          (pos[:, 1] >= lo[1]) & (pos[:, 1] < hi[1]))[0]
        if snap and len(inside):
            return pos[inside[np.argmax(pos[inside, 2])], :2]
        return lo + DWS / DHM

    g1 = cell_xy(int(heads[0][0].argmax()))
    g2 = cell_xy(int(heads[1][0].argmax()))
    p1 = cell_xy(int(heads[2][0].argmax()), snap=False)
    p2 = cell_xy(int(heads[3][0].argmax()), snap=False)
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    span = np.maximum(hi - lo, 1e-6)

    def inv(xy):
        return np.clip(2 * (np.asarray(xy) - lo) / span - 1, -1, 1)

    return np.concatenate([inv(g1), inv(g2), inv(p1), inv(p2)]).astype(
        np.float32)


# SSIM visual fold metric (second opinion vs the footprint proxy)
_fold_ssim = None


def garment_ssim(env):
    global _fold_ssim
    from metric_ssim import FoldSSIM
    from garment_folds_lib import reflect_fold, canonical_seq_for
    from metrics import particles
    from scene import TABLE_H
    if _fold_ssim is None:
        flat = env._flats[0]
        goal = flat.copy()
        for pt, n2 in canonical_seq_for("tshirt.obj", flat):
            d = (goal[:, :2] - pt) @ n2
            goal = reflect_fold(goal, pt, n2, d > 0)
        _fold_ssim = FoldSSIM(goal, TABLE_H)
    return _fold_ssim(particles(env.duo.cloth))


results = []
consec_fail = 0
for ep in range(args.episodes):
  # episode-level guard: a solver explosion (GenesisException escaping the
  # env's per-step guards, e.g. during reset or iron passes) must cost ONE
  # episode, not the whole eval
  try:
    obs, info = env.reset()
    if args.video and ep == 0:
        env.cam.start_recording()
    ep_r, steps = 0.0, 0
    garment_fq = {}
    garment_ss = {}
    while True:
        if dnets is not None:
            a = diffusion_action(env, dnets)
        elif actor is not None:
            with torch.no_grad():
                o = torch.as_tensor(obs).cuda().unsqueeze(0)
                a = actor.det(o).squeeze(0).cpu().numpy()
        else:
            a = np.random.uniform(-1, 1, 8).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        ep_r += r
        steps += 1
        garment_fq[info["garment"]] = info["fold_q"]
        garment_ss[info["garment"]] = garment_ssim(env)
        if args.video and ep == 0:
            env.cam.render()
        print(f"  ep {ep} step {steps}: r {r:+.2f} cov {info['coverage']:.2f} "
              f"fq {info['fold_q']:.2f} g{info['garment']} "
              f"stacked {info['stacked']}", flush=True)
        if term or trunc or steps >= 40:
            break
    burns = int(env.ironer.burn_count)
    results.append({"ep": ep, "reward": float(ep_r), "steps": steps,
                    "stacked": int(info["stacked"]), "burns": burns,
                    "final_fold_q": {str(k): float(v)
                                     for k, v in garment_fq.items()},
                    "final_ssim": {str(k): float(v)
                                   for k, v in garment_ss.items()}})
    print(f"EPISODE {ep}: reward {ep_r:.1f} stacked {info['stacked']} "
          f"burns {burns}", flush=True)
    consec_fail = 0
  except Exception as e:
    consec_fail += 1
    results.append({"ep": ep, "sim_crashed": True, "error": str(e)[:120],
                    "reward": None, "stacked": 0, "burns": 0,
                    "final_fold_q": {}, "final_ssim": {}})
    print(f"EPISODE {ep}: SIM CRASH ({str(e)[:80]})", flush=True)
    if consec_fail >= 2:
        print("two consecutive sim crashes — scene corrupted, stopping",
              flush=True)
        break
    # crash resilience: persist partial results after every episode
    try:
        with open(args.out + ".partial", "w") as f:
            json.dump({"episodes_done": ep + 1, "raw": results}, f, indent=1,
                      default=float)
    except OSError:
        pass                       # e.g. --out /dev/null on video-only runs
    if args.video and ep == 0:
        env.cam.stop_recording()
        mp4 = sorted(glob.glob("*_cam_0_*.mp4"), key=os.path.getmtime)[-1]
        os.rename(mp4, args.video)

ok_eps = [r for r in results if r.get("reward") is not None]
summary = {
    "policy": ("diffusion" if dnets is not None else "trained" if actor is not None else "random"),
    "mean_reward": float(np.mean([r["reward"] for r in ok_eps])) if ok_eps
    else None,
    "mean_stacked": float(np.mean([r["stacked"] for r in ok_eps]))
    if ok_eps else 0.0,
    "total_burns": int(sum(r["burns"] for r in ok_eps)),
    "sim_crashes": int(sum(1 for r in results if r.get("sim_crashed"))),
    "raw": results,
}
print(json.dumps({k: v for k, v in summary.items() if k != "raw"}, indent=1))
with open(args.out, "w") as f:
    json.dump(summary, f, indent=1)
print("EVAL DONE", flush=True)
