"""Train the fling policy with SAC over parallel worker processes.

Workers each own a Genesis scene and execute fling primitives; the learner
process holds the SAC nets and replay. Communication via multiprocessing pipes.

Usage: python train_fling.py --workers 5 --steps 2500 --out runs_fling
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np


def worker_proc(conn, seed, garment=None):
    os.environ.setdefault("GCOV_PREFIX", "/tmp/gcov")
    from env_laundry import LaundryEnv
    import traceback
    try:
        env = LaundryEnv(seed=seed, garment=garment or "tshirt4.obj")
        obs, info = env.reset(seed=seed)
    except Exception:
        print(f"[worker {seed}] build/reset crashed:\n{traceback.format_exc()}",
              flush=True)
        conn.send(("dead",))
        return
    conn.send(("reset", obs, info))
    while True:
        msg = conn.recv()
        if msg[0] == "step":
            try:
                o2, r, term, trunc, info = env.step(msg[1])
            except Exception as e:
                print(f"[worker {seed}] env crashed:\n{traceback.format_exc()}", flush=True)
                conn.send(("dead",))
                return
            done = term or trunc
            conn.send(("trans", o2, r, term, trunc, info))
            if done:
                try:
                    o0, i0 = env.reset()
                except Exception as e:
                    print(f"[worker {seed}] reset crashed:\n{traceback.format_exc()}", flush=True)
                    conn.send(("dead",))
                    return
                conn.send(("reset", o0, i0))
        elif msg[0] == "quit":
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2500)   # total fling primitives
    ap.add_argument("--out", type=str, default="runs_fling")
    ap.add_argument("--start_random", type=int, default=250)
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

    ctx = mp.get_context("spawn")
    conns, procs = [], []
    seed_counter = [1000]

    def spawn_worker(w):
        seed_counter[0] += 1
        pa, ch = ctx.Pipe()
        p = ctx.Process(target=worker_proc, args=(ch, seed_counter[0], args.garment),
                        daemon=True)
        p.start()
        return pa, p

    for w in range(args.workers):
        pa, p = spawn_worker(w)
        conns.append(pa)
        procs.append(p)

    cur_obs = [None] * args.workers
    for w, c in enumerate(conns):
        tag, obs, info = c.recv()
        cur_obs[w] = obs

    respawning = {}   # w -> awaiting first reset after respawn

    t0 = time.time()
    n_trans = 0
    ep_rew = [0.0] * args.workers
    ep_len = [0] * args.workers
    history = []
    covs_at_end, succ = [], []

    alive = [True] * args.workers
    while n_trans < args.steps and any(alive):
        # finish respawns: absorb the fresh worker's first reset (NON-blocking
        # — a fresh laundry worker needs ~10 min to build; never stall the fleet)
        for w in list(respawning):
            try:
                if not conns[w].poll():
                    if not procs[w].is_alive():
                        raise EOFError
                    continue
                tag, obs, info = conns[w].recv()
                cur_obs[w] = obs
                alive[w] = True
                del respawning[w]
                print(f"[learner] worker {w} respawned", flush=True)
            except (EOFError, OSError):
                conns[w], procs[w] = spawn_worker(w)
        # dispatch one action to each live worker
        acts = [None] * args.workers
        for w in range(args.workers):
            if not alive[w]:
                continue
            if n_trans < args.start_random:
                a = np.random.uniform(-1, 1, ACT).astype(np.float32)
            else:
                a = agent.act(cur_obs[w])
            acts[w] = a
            conns[w].send(("step", a))
        # collect
        for w in range(args.workers):
            if not alive[w]:
                continue
            try:
                # bounded wait: a legit step (incl. an end-of-episode reset
                # with 3 crumples + pick) takes <= ~15 min; a worker silent
                # for 20 min is HUNG (solver spin) — kill it, don't stall
                # the whole fleet on a blocking recv
                deadline = time.time() + 1200
                msg = None
                while time.time() < deadline:
                    if conns[w].poll(5):
                        msg = conns[w].recv()
                        break
                    if not procs[w].is_alive():
                        break
                if msg is None:
                    print(f"[learner] worker {w} hung >20min; killing",
                          flush=True)
                    procs[w].kill()
                    msg = ("dead",)
            except (EOFError, OSError):
                msg = ("dead",)
            if msg[0] == "dead":
                print(f"[learner] worker {w} died; respawning", flush=True)
                conns[w], procs[w] = spawn_worker(w)
                respawning[w] = True
                alive[w] = False
                continue
            tag, o2, r, term, trunc, info = msg
            done = term or trunc
            buf.add(cur_obs[w], acts[w], r, o2, float(term))
            n_trans += 1
            ep_rew[w] += r
            ep_len[w] += 1
            if done:
                covs_at_end.append(info.get("fold_q", 0.0))
                succ.append(1.0 if info.get("stacked", 0) >= 2 else 0.0)
                history.append({"t": time.time() - t0, "n": n_trans,
                                "ep_rew": ep_rew[w], "ep_len": ep_len[w],
                                "coverage": float(info["coverage"]), "stacked": int(info.get("stacked", 0)), "fold_q": float(info.get("fold_q", 0.0))})
                ep_rew[w] = 0.0
                ep_len[w] = 0
                try:
                    msg2 = conns[w].recv()
                except (EOFError, OSError):
                    msg2 = ("dead",)
                if msg2[0] == "dead":
                    print(f"[learner] worker {w} died on reset; respawning", flush=True)
                    conns[w], procs[w] = spawn_worker(w)
                    respawning[w] = True
                    alive[w] = False
                    continue
                cur_obs[w] = msg2[1]
            else:
                cur_obs[w] = o2
        # gradient steps (replay ratio ~4 per collected transition)
        if buf.n > 128:
            for _ in range(4 * args.workers):
                agent.update(buf.sample(256))
        if n_trans % 20 < 2:
            last = covs_at_end[-20:] or [0]
            print(f"[{time.time()-t0:6.0f}s] actions {n_trans:5d} "
                  f"| recent end fold_q {np.mean(last):.3f} "
                  f"| recent success {np.mean(succ[-20:] or [0]):.2f}", flush=True)
            with open(os.path.join(args.out, "history.json"), "w") as f:
                json.dump(history, f, default=float)
            import torch
            torch.save({"actor": agent.actor.state_dict()},
                       os.path.join(args.out, "actor.pt"))

    import torch
    torch.save({"actor": agent.actor.state_dict()},
               os.path.join(args.out, "actor_final.pt"))
    print(f"DONE {n_trans} flings in {time.time()-t0:.0f}s; "
          f"final success rate (last 30 eps): {np.mean(succ[-30:] or [0]):.2f}")
    for w, c in enumerate(conns):
        if alive[w]:
            try:
                c.send(("quit",))
            except OSError:
                pass


if __name__ == "__main__":
    main()
