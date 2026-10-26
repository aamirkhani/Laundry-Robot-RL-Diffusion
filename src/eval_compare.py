"""Compare untangling policies: trained SAC actor vs corner heuristic vs random.

Runs N tangled towels per policy in one persistent scene; reports end coverage
after up to K flings each. Usage:
    python eval_compare.py --episodes 10 --actor runs_fling2/actor_final.pt
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--actor", type=str, default="runs_fling2/actor_final.pt")
ap.add_argument("--max_flings", type=int, default=5)
ap.add_argument("--denoiser", type=str, default="denoiser.pt")
ap.add_argument("--only_diffusion", type=str, default="",
                help="tag:path[,tag:path] — compare specific denoiser checkpoints")
ap.add_argument("--garment", type=str, default="")
ap.add_argument("--out", type=str, default="eval_compare.json")
args = ap.parse_args()

from scene import build_scene, settle, spawn_tangled, hold_home, HOME_L, HOME_R
from primitives import Duo
from metrics import particles, coverage
from baseline import pick_fling_points

scene, left, right, cloth, cam = build_scene(n_grid=31, seed=123,
                                             garment=args.garment or None)
scene.build()
left.set_dofs_position(HOME_L)
right.set_dofs_position(HOME_R)
hold_home(left, right)
settle(scene, 100)
flat = particles(cloth).copy()
duo = Duo(scene, left, right, cloth, flat_pos=flat)

actor = None
try:
    import os
    import torch
    from sac import Actor
    from env import GRID, WS
    if os.path.exists(args.actor):
        ck = torch.load(args.actor, map_location="cuda")
        actor = Actor(65, 4).cuda()
        actor.load_state_dict(ck["actor"])
        actor.eval()
        print(f"loaded actor {args.actor}")
except Exception as e:
    print("no actor:", e)


def obs_of(pos):
    from env import GRID, WS
    hm = np.zeros((GRID, GRID), dtype=np.float32)
    ij = np.clip(((pos[:, :2] + WS) / (2 * WS) * GRID).astype(int), 0, GRID - 1)
    for (i, j), z in zip(ij, pos[:, 2]):
        hm[i, j] = max(hm[i, j], z)
    return np.concatenate([hm.flatten() / 0.08, [coverage(pos)]]).astype(np.float32)


def act_to_particles(a, pos):
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    idx = []
    for m in range(2):
        t = (a[2 * m:2 * m + 2] + 1) / 2
        raw = lo + t * (hi - lo)
        idx.append(int(np.argmin(((pos[:, :2] - raw) ** 2).sum(1))))
    return idx


HM16 = 16
WS16 = 0.35


def obs16(pos):
    from scene import TABLE_H
    hm = np.zeros((HM16, HM16), dtype=np.float32)
    ij = np.clip(((pos[:, :2] + WS16) / (2 * WS16) * HM16).astype(int), 0, HM16 - 1)
    for (i, j), z in zip(ij, pos[:, 2] - TABLE_H):
        hm[i, j] = max(hm[i, j], z)
    return np.concatenate([hm.flatten() / 0.08, [coverage(pos)]]).astype(np.float32)


denoiser = None
try:
    import os as _os
    if _os.path.exists(args.denoiser):
        import torch
        _ck = torch.load(args.denoiser, map_location="cuda")
        if any(k.startswith("heads.") for k in _ck["net"]):
            from train_denoiser2 import Denoiser4 as _DNC
        else:
            from train_denoiser import Denoiser as _DNC
        denoiser = _DNC().cuda()
        denoiser.load_state_dict(_ck["net"])
        denoiser.eval()
        print(f"loaded denoiser {args.denoiser}")
except Exception as e:
    print("no denoiser:", e)


def cell_to_particle(cell, pos):
    ci, cj = cell // HM16, cell % HM16
    lo_x = -WS16 + ci * 2 * WS16 / HM16
    lo_y = -WS16 + cj * 2 * WS16 / HM16
    inside = np.where((pos[:, 0] >= lo_x) & (pos[:, 0] < lo_x + 2 * WS16 / HM16) &
                      (pos[:, 1] >= lo_y) & (pos[:, 1] < lo_y + 2 * WS16 / HM16))[0]
    if len(inside):
        return int(inside[np.argmax(pos[inside, 2])])
    cc = np.array([lo_x + WS16 / HM16, lo_y + WS16 / HM16])
    return int(np.argmin(((pos[:, :2] - cc) ** 2).sum(1)))


class DiffusionChooser:
    """Reverse-process rollout: the denoiser proposes the inverse un-fold
    grasps. Hybrid mode = coverage validator (paper's denoiser+validator):
    if the last proposal failed to improve coverage, take the next-best cells."""

    def __init__(self, hybrid=False):
        self.hybrid = hybrid
        self.reset()

    def reset(self):
        self.last_cov = None
        self.rank = 0

    def __call__(self, pos, rng):
        import torch
        cov = coverage(pos)
        if self.hybrid and self.last_cov is not None and cov < self.last_cov + 0.01:
            self.rank = min(self.rank + 1, 4)
        else:
            self.rank = 0
        self.last_cov = cov
        net = getattr(self, "net", None) or denoiser
        with torch.no_grad():
            o = torch.as_tensor(obs16(pos)).cuda().unsqueeze(0)
            heads = net(o)
            l1, l2 = heads[0], heads[1]
            c1 = int(torch.topk(l1[0], 5).indices[self.rank])
            c2 = int(torch.topk(l2[0], 5).indices[self.rank])
        return cell_to_particle(c1, pos), cell_to_particle(c2, pos)


def run_policy(name, chooser, seeds):
    covs, flings_used = [], []
    for s in seeds:
        getattr(chooser, "reset", lambda: None)()
        rng = np.random.default_rng(s)
        duo.release_all()
        duo.home(seconds=0.4)
        cloth.set_particles_pos(flat + [0, 0, 0.02])
        cloth.set_particles_vel(np.zeros_like(flat))
        settle(scene, 60)
        spawn_tangled(scene, cloth, rng, settle_steps=200)
        used = 0
        for f in range(args.max_flings):
            pos = particles(cloth)
            if coverage(pos) > 0.74:
                break
            try:
                duo.ensure_centered()
                i, j = chooser(particles(cloth), rng)
                p = particles(cloth)
                if p[i, 0] > p[j, 0]:
                    i, j = j, i
                duo.fling(None, None, particles_lr=(i, j))
            except Exception as e:
                print(f"  [!] sim crashed on seed {s}: {e}; ending episode", flush=True)
                break
            used += 1
        c = coverage(particles(cloth))
        covs.append(c)
        flings_used.append(used)
        print(f"  {name} seed {s}: coverage {c:.3f} ({used} flings)", flush=True)
    return covs, flings_used


def choose_heuristic(pos, rng):
    return pick_fling_points(pos, flat)


def choose_random(pos, rng):
    return list(rng.integers(0, len(pos), 2))


def choose_actor(pos, rng):
    import torch
    with torch.no_grad():
        o = torch.as_tensor(obs_of(pos)).cuda().unsqueeze(0)
        a = torch.tanh(actor(o)[0]).squeeze(0).cpu().numpy()
    return act_to_particles(a, pos)


seeds = list(range(200, 200 + args.episodes))
results = {}
policies = []
if args.only_diffusion:
    import torch as _t
    for tag, path in [p.split(":") for p in args.only_diffusion.split(",")]:
        _ck2 = _t.load(path, map_location="cuda")["net"]
        if any(k.startswith("heads.") for k in _ck2):
            from train_denoiser2 import Denoiser4 as _D
        else:
            from train_denoiser import Denoiser as _D
        _n = _D().cuda()
        _n.load_state_dict(_ck2)
        _n.eval()
        ch = DiffusionChooser(hybrid=False)
        ch.net = _n
        policies.append((tag, ch))
else:
    policies = [("heuristic", choose_heuristic), ("random", choose_random)]
    if actor is not None:
        policies.append(("rl_actor", choose_actor))
    if denoiser is not None:
        policies.append(("diffusion", DiffusionChooser(hybrid=False)))
        policies.append(("diffusion_hybrid", DiffusionChooser(hybrid=True)))
for name, chooser in policies:
    print(f"=== {name} ===", flush=True)
    try:
        covs, used = run_policy(name, chooser, seeds)
    except Exception as e:
        print(f"[!] policy {name} aborted: {e}", flush=True)
        continue
    results[name] = {"coverages": covs, "flings": used,
                     "mean_cov": float(np.mean(covs)),
                     "success@0.74": float(np.mean([c > 0.74 for c in covs]))}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "coverages"}
                  for k, v in results.items()}, indent=1))
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print(f"saved {args.out}")
