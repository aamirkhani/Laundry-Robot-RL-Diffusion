# Staged untangle → two-fold: RL vs diffusion (2026-09-01/02)

Task: a garment spawns TANGLED on the table (3–5 random reflection folds,
applied sequentially with damped settling). The robots must (1) untangle it
via two-arm flings until coverage > 0.72, then (2) fold it into the canonical
TWO-FOLD state (left-over-right, then hem-to-collar). Scene: 2.2 m × 1.6 m
table at 0.75 m, glass enclosure, two face-to-face Frankas + third ironing
arm, cotton-look cloth (bending 1.4e-2, air 0.006).

Metric: goal_err = mean per-particle distance to the canonical two-fold
configuration after best 2D rigid alignment (Procrustes). Lower is better.
An untouched FLAT garment scores ~0.15–0.16; the tangled start varies.

## Headline

| method (t-shirt) | mean goal_err ↓ | notes |
|---|---|---|
| diffusion (scramble-inversion) | 0.143 | zero robot data for training; 1 sim-crash ep excluded |
| scripted oracle (privileged) | 0.142 | knows the tangle's true inverse + canonical fold keypoints |
| SAC @ ~400 actions | 0.141 | 1 sim-crash ep excluded |
| SAC @ 1,000 actions (6 h) | 0.137 | 1 sim-crash ep excluded |

**Three-way statistical tie on the t-shirt** (same 6 tangle seeds per mode;
sim-crash episodes excluded — 1 for diffusion, 1 for SAC, 0 for oracle).
An untouched flat shirt scores ~0.15, so on this task the metric saturates
near "flat": nobody — not even the privileged oracle executing known-perfect
keypoints through real arms — produces two-folds that beat flat by a wide
margin. The bottleneck is fold-execution physics (rounded creases, placement
slack, occasional explosions), not the policy. This differs from towels,
where diffusion separated decisively (0.76 vs 0.42 coverage).

| method (pants) | mean goal_err ↓ |
|---|---|
| scripted oracle | **0.115** |
| diffusion | 0.151 |

Pants is where policy quality shows: the oracle cleanly beats diffusion,
whose pants fold-denoiser place-heads are weak (12–19% val accuracy — the
long symmetric legs alias the grasp/place corners). More data or a
leg-symmetry-aware loss is the known fix.

The 0.06 "success" threshold used during development proved physically
unreachable for arm-executed folds — even the oracle lands at ~0.14. The
canonical goal is a perfect mirror-image stack; real folds have rounded
creases and placement slack. Report goal_err directly instead.

## What it took to get here (the day's real findings)

1. **Material stability limit (the big one).** The first "cotton" params
   (bending 8e-3, air 0.02) explode the PBD solver whenever any fold lands
   on an already-folded stack. Consequence discovered by auditing training
   data: **100% of the folded-goal scramble dataset was garbage** — and it
   passed `isfinite` checks, because exploded particles clamp at the PBD
   upper bound instead of going NaN. Audit rule: any 16×16 heightmap cell
   > 0.30 m ⇒ exploded sample. Stress-tested replacement (bending 1.4e-2,
   air 0.006) survives the two-fold goal + 3 stacked scramble folds.
2. **Tangle spawning had to become sequential.** Teleporting a multi-fold
   tangle in one shot explodes even with the stable material; applying one
   fold at a time with damped settling (the data collectors' recipe) is
   stable. `spawn_tangled` was rewritten accordingly.
3. **Oracle reward for RL (the user's hack).** Since we synthesize the
   tangle, the true inverse un-fold grasp points are free. `spawn_tangled`
   returns the last fold's flap corners; the staged env pays a dense bonus
   (exp(-dist/8cm)) for first-fling grasps near them.
4. **Refined denoiser loss**: Gaussian soft targets over grasp/place cells,
   coverage-weighted samples (near-goal states get 2.5× weight), and
   unordered-pair symmetry (min over head-swap). Deeper 4-conv trunk.

## Models trained (clean data, explosion-filtered)

| model | data | val within-1.5-cells (chance ≈ 3.5%) |
|---|---|---|
| untangle_tshirt.pt | 1,327 pairs | g1 46% g2 47% / p1 37% p2 33% |
| untangle_pants.pt | 1,205 pairs | (comparable) |
| fold_tshirt.pt | 1,176 pairs | g1 42% g2 45% / p1 41% p2 30% |
| fold_pants.pt | 1,238 pairs | g1 12% g2 14% / p1 12% p2 19% — the long symmetric legs alias the grasp corners |

## RL (SAC over fling/fold primitives, staged reward)

- Env: `env_staged.py` — obs 259-d (16×16 z-relative heightmap + coverage +
  goal_err + phase), action 8-d, phase-switched primitives, oracle bonus,
  8 actions/episode cap. Trainer: `train_staged.py` (10 parallel Genesis
  workers, crash-respawn).
- Wall-clock reality: one primitive action ≈ 30–75 s even at 10 workers
  (sequential tangle reset + real arm motion). ~400 actions took ~4.5 h.
  Diffusion's entire training set (armless teleport scrambles) collected in
  about the same time — the ~50× cost gap of previous rounds persists.
- Training-time success (goal_err < 0.06) stayed 0.00 — expected, given the
  threshold is unreachable (see above); end-episode goal_err hovered ~0.14
  before learning could bite.
- SAC@400 per-episode (t-shirt, same seeds as above): 0.117, 0.168, 0.112,
  0.163, crash, 0.146 — indistinguishable from diffusion/oracle. At ~400
  mostly-random actions this is "fling until flat", which the saturated
  metric rewards as much as anyone's folds. An extended overnight run
  (runs_staged2, 1,000 actions) tests whether the oracle-shaped reward
  eventually produces real fold behavior. RESULT: it did not — 1,000
  actions over 6 h plateaued at training goal_err ~0.135-0.14 and evaluated
  at 0.137 (per-episode 0.137, 0.126, 0.143, crash, 0.140, 0.139), within
  noise of SAC@400, diffusion, and the oracle. 2.5x more robot experience
  bought nothing the saturated metric could measure.

## Sim-crash accounting

Episodes with final coverage < 0.05 are solver explosions during arm
interaction, not scoreable policy outcomes: diffusion t-shirt 1/6, SAC@400
1/6, SAC@1000 1/6, oracle t-shirt 0/6, pants 0/12. Excluded from means.

## Artifacts

- `eval_staged_tshirt.json`, `eval_staged_pants.json`,
  `eval_staged_tshirt_sac.json` — raw per-episode numbers
- `staged_tshirt.mp4`, `staged_pants.mp4` — full eval recordings
  (table + glass + 3 arms visible, cotton-look cloth)
- `untangle_*.pt`, `fold_*.pt`, `runs_staged/actor.pt`, `runs_staged2/actor_final.pt` — checkpoints
- `data_scramble5/` — clean scramble data (0% exploded)
- `env_staged.py`, `train_staged.py`, `eval_staged.py` — staged stack
