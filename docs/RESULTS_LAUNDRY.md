# Laundry v2: full pick→uncrumble→iron→fold→stack, RL vs diffusion (2026-09-04/05)

Task: 3 crumpled garments start in a bucket hanging OUTSIDE the table's left
rim (body fully off-table, mounted to the rim, reachable through a gap in the
glass wall). Per garment: scripted pick onto the table → POLICY uncrumples
(flings) → automatic single iron pass when flat (heat model; dwelling burns
scorch marks and is penalized) → POLICY folds → scripted stack. Episode = all
3 garments, ≤8 policy actions each.

Garments: TWO-LAYER 4×-vertex meshes per user spec — tshirt4.obj (3,330
verts, sim 6,592 particles @ psize 0.0072) and pants4.obj (3,306 verts,
4,357 @ 0.008). The ironing arm stands fully ON the tabletop.

Fold metric (non-saturating): fold_q = 0.5·area-ratio-gaussian(target 0.27)
+ 0.3·rectangularity + 0.2·thinness. Random-policy episode-end baseline
≈ 0.27–0.30.

## RL (SAC, CNN encoder, ~1,700 cumulative arm actions over ~3 days)

Model: conv encoder over 32×32 heightmap + twin critics (sac_cnn.py) — the
"more complex model" spec. Trained with oracle-free phase-shaped reward,
warm-started across restarts.

Final eval, 6 fresh bucket episodes (seed 900), vs random on SAME seeds:

| policy | mean reward | stacks (of 18 garments) | burns |
|---|---|---|---|
| SAC trained | 2.3 | 1 | 0 |
| random | 2.6 (excl. one −160 burn-catastrophe ep) | 1 | 108 (one episode) |

**Verdict: after ~1,700 real arm actions, the RL policy is statistically
indistinguishable from random on task outcomes.** Two positive nuances, both
under-powered: episode-end fold_q during late training trended 0.36–0.46 vs
the 0.27 baseline, and the trained policy logged zero burns while random hit
one catastrophic burn episode. Neither survives n=6 scrutiny.

Training-side reality that dominated the 3 days: each policy action costs
30–100 s of wall-clock (real arm motion + 20k-particle sim). The run
survived — and required — crash-respawn, hang-detection (kill after 20 min
silent; fired 3+ times), JSON-serialization fixes, and warm-restarts across
4 process restarts. RL at this fidelity is an infrastructure problem first.

## Diffusion (scramble-inversion, zero arm actions for training)

Data: armless teleport scrambles, ~40 min per collector ×6. Uncrumple data
clean (0–1% exploded); folded-goal data on the 4× two-layer meshes explodes
often (tshirt 64–99%(!), pants 20–29%) — the two-fold goal state is at the
solver's stability edge at this resolution. After explosion-filtering:
uncrumple 497/491 pairs, fold 125 (tshirt) / 342 (pants) pairs.

Denoisers (best of 4 seeds — training variance is large at these data
sizes): tshirt fold grasp-heads 83/67% within 1.5 cells (chance 3.5%),
pants fold 65/62%, place-heads 33–50%.

Same-seed laundry eval (6 episodes each, seed 900):

| policy | mean reward | stacks/18 | burns | mean fold_q |
|---|---|---|---|---|
| SAC trained (~1,700 arm actions) | 2.3 | 1 | 0 | 0.328 |
| random | −24.5 (2.6 excl. burn ep) | 1 | 108 | 0.334 |
| diffusion t-shirt | −0.4 | 0 | 0 | 0.305 |
| diffusion pants | −0.9 | 0 | 0 | 0.283 |

## Combined verdict — honest and useful

**On the full laundry loop, nothing beats random yet.** Fold quality sits at
the ~0.30 baseline for all policies; each stacked 1-or-0 garments of 18. The
one real difference: both non-random policies produced ZERO burns while
random suffered one 108-burn catastrophe (−160 reward) — structure helps
safety before it helps skill.

Why diffusion — which matched a privileged oracle on the staged two-fold
task — fails here: distribution shift. The denoisers train on teleport-
scramble states, but laundry states come from bucket-crumple → arm-pick →
drop, which look different in the heightmap; and the fold denoisers are
data-starved because the two-fold goal state on 4× two-layer meshes sits at
the PBD solver's stability edge (64–99% of folded-goal scrambles explode).
Why RL fails: ~1,700 actions is tiny for a CNN policy on a 3-garment
sequential task, and each action costs 30–100 s — the ~50× cost gap versus
armless diffusion data means matching diffusion's data volume would take
months.

What would move the needle, in order: (1) a cloth engine stable enough to
collect folded-goal data at full resolution (the user's planned physics
rewrite directly addresses this), (2) collect scrambles from arm-dropped
states instead of teleports (kills the distribution shift), (3) 10× more
fold-denoiser data.

## Cost accounting

| | RL | diffusion |
|---|---|---|
| training interactions | ~1,700 arm actions | 0 |
| wall-clock to train | ~3 days (with babysitting) | ~2 h collect + minutes train |
| infrastructure needed | respawn+hang-watchdog trainer | none |

## Artifacts

- `eval_rl_final.json`, `eval_random_final.json`, `eval_diff_tshirt.json`,
  `eval_diff_pants.json` — raw episodes
- `laundry_rl_final.mp4`, `laundry_diff_tshirt.mp4`,
  `laundry_diff_pants.mp4` — full-loop videos
- `runs_laundry2/actor_final.pt`, `uncrumple_*.pt`, `fold_*4.pt`
- `env_laundry.py`, `train_laundry2.py`, `eval_laundry.py`,
  `scene_laundry.py`, `primitives_laundry.py`
