"""Laundry-v2 trainer, event-driven collect (fixes the v1 throughput wall).

v1 failure mode: sequential blocking recv per worker — one slow episode
(iron passes on 6.6k-particle cloth take minutes) stalls the whole fleet;
observed 13 h with ~0 transitions. Here workers are multiplexed with
multiprocessing.connection.wait(): whichever worker finishes its step is
served immediately and re-dispatched; a worker silent past HANG_S is killed
and respawned without blocking anyone.
"""
import argparse
import json
import os
import time

import numpy as np
import multiprocessing as mp
from multiprocessing.connection import wait as conn_wait

HANG_S = 1500.0


def worker_proc(conn, seed, garment=None):
    os.environ.setdefault("GCOV_PREFIX", "/tmp/gcov")
    from env_laundry import LaundryEnv
    env = LaundryEnv(seed=seed, garment=garment or "tshirt4.obj")
    obs, info = env.reset()
    conn.send(("reset", obs, info))
    while True:
        msg = conn.recv()
        if msg[0] == "close":
            break
        try:
            o2, r, term, trunc, info = env.step(msg[1])
            done = term or trunc
            conn.send(("step", o2, r, term, trunc, info))
            if done:
                obs, info = env.reset()
                conn.send(("reset", obs, info))
        except Exception:
            import traceback
            print(f"[worker {seed}] env crashed:", flush=True)
            traceback.print_exc()
            try:
                conn.send(("dead",))
            except Exception:
                pass
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", type=str, default="runs_laundry")
    ap.add_argument("--start_random", type=int, default=40)
    ap.add_argument("--garment", type=str, default=None)
    ap.add_argument("--init_actor", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from sac_cnn import SACCNN, Replay
    OBS, ACT = 32 * 32 + 6, 8
    agent = SACCNN(32, 6, ACT)
    if args.init_actor and os.path.exists(args.init_actor):
        import torch
        ck = torch.load(args.init_actor, map_location="cuda")
        agent.actor.load_state_dict(ck["actor"])
        print(f"[learner] warm-started actor from {args.init_actor}", flush=True)
    buf = Replay(OBS, ACT)
    rp = os.path.join(args.out, "replay.npz")
    if os.path.exists(rp):
        d = np.load(rp)
        n = int(d["n"])
        buf.o[:n], buf.a[:n], buf.r[:n] = d["o"][:n], d["a"][:n], d["r"][:n]
        buf.o2[:n], buf.d[:n] = d["o2"][:n], d["dn"][:n]
        buf.n = n
        buf.i = n % buf.cap
        print(f"[learner] restored replay: {n} transitions", flush=True)

    ctx = mp.get_context("spawn")

    def spawn_worker(w):
        pc, cc = ctx.Pipe()
        p = ctx.Process(target=worker_proc,
                        args=(cc, 1000 + w, args.garment), daemon=True)
        p.start()
        return pc, p

    conns, procs = {}, {}
    for w in range(args.workers):
        conns[w], procs[w] = spawn_worker(w)

    cur_obs = {}
    last_act = {}
    last_heard = {w: time.time() for w in range(args.workers)}
    awaiting = {w: "reset" for w in range(args.workers)}   # reset|step|idle

    t0 = time.time()
    n_trans = 0
    ep_rew = {w: 0.0 for w in range(args.workers)}
    ep_len = {w: 0 for w in range(args.workers)}
    history, covs_at_end, succ = [], [], []

    import torch

    def dispatch(w):
        if n_trans < args.start_random:
            a = np.random.uniform(-1, 1, ACT).astype(np.float32)
        else:
            a = agent.act(cur_obs[w])
        last_act[w] = a
        conns[w].send(("step", a))
        awaiting[w] = "step"

    def respawn(w):
        try:
            procs[w].kill()
        except Exception:
            pass
        conns[w], procs[w] = spawn_worker(w)
        awaiting[w] = "reset"
        last_heard[w] = time.time()
        ep_rew[w], ep_len[w] = 0.0, 0

    while n_trans < args.steps:
        conn_to_w = {conns[w]: w for w in conns}
        ready = conn_wait(list(conn_to_w), timeout=30)
        now = time.time()
        for c in ready:
            w = conn_to_w[c]
            try:
                msg = c.recv()
            except (EOFError, OSError):
                print(f"[learner] worker {w} pipe died; respawning", flush=True)
                respawn(w)
                continue
            last_heard[w] = now
            if msg[0] == "dead":
                print(f"[learner] worker {w} died; respawning", flush=True)
                respawn(w)
                continue
            if msg[0] == "reset":
                cur_obs[w] = msg[1]
                dispatch(w)
                continue
            _, o2, r, term, trunc, info = msg
            done = term or trunc
            buf.add(cur_obs[w], last_act[w], r, o2, float(term))
            cur_obs[w] = o2
            n_trans += 1
            ep_rew[w] += r
            ep_len[w] += 1
            if done:
                covs_at_end.append(float(info.get("fold_q", 0.0)))
                succ.append(1.0 if info.get("stacked", 0) >= 2 else 0.0)
                history.append({"t": now - t0, "n": n_trans,
                                "ep_rew": float(ep_rew[w]),
                                "ep_len": int(ep_len[w]),
                                "coverage": float(info["coverage"]),
                                "stacked": int(info.get("stacked", 0)),
                                "fold_q": float(info.get("fold_q", 0.0))})
                ep_rew[w], ep_len[w] = 0.0, 0
                awaiting[w] = "reset"      # worker auto-resets, will send obs
            else:
                dispatch(w)
            # train on every collected transition
            if buf.n > 128:
                for _ in range(4):
                    agent.update(buf.sample(256))
            if n_trans % 20 == 0:
                last = covs_at_end[-20:] or [0]
                print(f"[{now-t0:6.0f}s] actions {n_trans:5d} "
                      f"| recent end fold_q {np.mean(last):.3f} "
                      f"| recent success {np.mean(succ[-20:] or [0]):.2f}",
                      flush=True)
                with open(os.path.join(args.out, "history.json"), "w") as f:
                    json.dump(history, f, default=float)
                torch.save({"actor": agent.actor.state_dict()},
                           os.path.join(args.out, "actor.pt"))
        # hang sweep (never blocks)
        for w in list(conns):
            if awaiting[w] != "idle" and now - last_heard[w] > HANG_S:
                print(f"[learner] worker {w} hung "
                      f"{int(now - last_heard[w])}s; respawning", flush=True)
                respawn(w)

    for w in conns:
        try:
            conns[w].send(("close",))
        except Exception:
            pass
    torch.save({"actor": agent.actor.state_dict()},
               os.path.join(args.out, "actor_final.pt"))
    print(f"DONE {n_trans} actions in {time.time()-t0:.0f}s; "
          f"final success rate (last 30 eps): {np.mean(succ[-30:] or [0]):.2f}")


if __name__ == "__main__":
    main()
