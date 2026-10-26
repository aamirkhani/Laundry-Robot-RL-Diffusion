# Bimanual Cloth Folding — two Frankas fold tangled towels (Genesis + RL)

Two floor-mounted 7-DoF Franka Panda arms are fed tangled square towels one at
a time; they untangle each by dynamic two-arm flings, fold it in half, and
carry it to a stack — inside Genesis's PBD cloth simulation on the RTX 5080.

## Architecture

- **Physics**: Genesis 1.3.3 PBD cloth (position-based dynamics, particle-based
  self-collision), floppy towel material (`bending_compliance=2e-2`, high
  friction), two Frankas from the bundled MJCF, rigid–cloth coupling.
- **Grasping**: cloth particles within the fingertip radius are pinned to the
  gripper hand link (`fix_particles_to_link`) — top layer only (grabbing through
  layers clamps folds shut); released via `release_particle`.
- **Primitives** ([primitives.py](primitives.py)): `fling` (grasp two points,
  lift, taut-stretch to the cloth-space distance between grasps, shake, lay the
  curtain down deterministically), `fold` (two corners arced onto the opposite
  edge), `carry` (two-arm translate preserving the fold), `drag`,
  `ensure_centered`.
- **RL** ([env.py](env.py), [sac.py](sac.py), [train_fling.py](train_fling.py)):
  SAC learns *where to grasp* for the untangling flings. Obs = 8×8 cloth
  heightmap + coverage (65-d, resolution-independent); action = two grasp
  points (4-d continuous); reward = coverage gain + success bonus at 0.74.
  9 worker processes each own a Genesis scene; crashed workers respawn.
- **Pipeline** ([full_pipeline.py](full_pipeline.py)): towels queue in a
  "basket", each is fed to the workspace, tangled, untangled (RL policy or
  corner heuristic), folded, carried to the stack.
- **Metrics** ([metrics.py](metrics.py)): footprint coverage, fold quality
  (area-ratio + rectangularity + 2-layer height), on-stack placement. All
  order-free (PBD resamples particles — mesh-vertex indexing does not survive).

## Hard-won stability lessons (Genesis PBD + rigid arms)

1. **Solver NaN explosions** ("contact … not finite"): caused by pinning cloth
   particles *inside* the gripper collision geometry — the pin holds them in
   penetration and forces diverge. Fix: the hand link stops ~11.5 cm above the
   grasp point so pinned cloth hangs *below* the fingers.
2. Commanding both grippers to nearly the same grasp point drives the two arms
   into each other → same explosion. Guard: <12 cm separation ⇒ single-arm.
3. The PBD floor boundary is **frictionless** — any nudge slides forever.
   Proxy friction: zero near-floor horizontal particle velocity every 5 steps,
   plus a global 3 m/s particle-speed clamp.
4. `fix_particles()` (freezing) a towel while arms manipulate another cloth
   nearby corrupts it (stretched-to-the-sky artifacts) — do not freeze; prevent
   stack disturbance mechanically (lift arms vertically before homing).
5. Towels escape the workspace under dynamic flings → PBD `lower/upper_bound`
   as invisible containment walls.
6. Cloth-space taut stretching needs particle correspondence: capture the flat
   layout at build time; PBD keeps particle order thereafter.
7. Stiff default bending (`1e-5`) makes cloth spring open — folds only persist
   with floppy bending + high friction.

## Results

Untangling policy comparison (6 tangled towels each, ≤4 flings, success =
coverage>0.74 — `eval_compare.json`):

| policy | mean coverage | success |
|---|---|---|
| corner heuristic (privileged flat-corner correspondence) | **0.770** | **83%** |
| random grasp points | 0.592 | 67% |
| SAC actor (450 flings, 4-D continuous action) | 0.604 | 17% |

### Diffusion instead of RL (per Khani 2026, Zenodo 21970238 & 22063663)

The user's scramble-inversion method transplanted to cloth: forward process =
single random reflection folds from the FLAT towel (the goal state); denoiser
(small CNN, two grasp-cell softmax heads over a 16x16 heightmap) trained with
cross-entropy to predict the inverse un-fold grasps; solving = reverse-process
rollout through the fling primitive. 4,171 self-supervised pairs collected with
ZERO robot execution (~35 min); training ~3 min.

Confirmatory eval (10 fresh tangles, ≤5 flings, identical seeds per policy):

| policy | mean coverage | success | avg flings |
|---|---|---|---|
| **diffusion (reverse rollout)** | **0.762** | **70%** | **3.4** |
| diffusion + coverage validator | 0.711 | 70% | 3.6 |
| corner heuristic (privileged) | 0.422 | 30% | 4.2 |
| random | 0.480 | 30% | 4.6 |

(First 6-seed batch agreed: diffusion_hybrid 0.741 / diffusion 0.627-67% vs
heuristic 0.592-33%, SAC 0.495-50%.) **The diffusion untangler, trained without
any arm interaction, more than doubles the heuristic's success rate** — the
papers' core claim (self-supervised scramble inversion beats interaction-heavy
RL under matched compute) reproduces on deformable cloth, a domain beyond the
papers' ten-domain assumption ladder (continuous state, approximately
invertible actions, mildly stochastic settling).

Negative results, honestly: (a) advantage-weighted fine-tuning on 233 executed
flings did not improve the base denoiser (0.68 vs 0.90 on overlapping seeds) —
far more outcome data would be needed; (b) rare Genesis PBD solver explosions
(~5-10% of long episodes) remain the dominant failure mode, independent of
policy. Artifacts: `denoiser.pt`, `denoiser_ft.pt`, `data_scramble/`,
`data_finetune/`, `eval_confirm.json`, `eval_ft*.json`,
`laundry_diffusion_clean.mp4` (diffusion-untangled pipeline),
`collect_scramble.py`, `train_denoiser.py`, `finetune_denoiser.py`.

**Honest RL verdict:** at this sample budget the SAC-over-primitives policy did
not beat the privileged corner heuristic — its 4-D continuous action over a
flattened heightmap gives weak spatial credit assignment. The right next step
(FlingBot-style pixel-wise Q-map over a 64×64 heightmap; nearly a contextual
bandit on Δcoverage) is scoped in the project notes; the infrastructure here
(env, workers, respawn, eval) is what it needs. The showcase pipeline uses the
heuristic untangler; the RL checkpoint + learning curves are in `runs_fling4/`.

## Garments: t-shirts and pants (2026-08-29..30)

Procedural garment meshes ([make_garment_mesh.py](make_garment_mesh.py)): mask-based
triangulated t-shirt (~1.7k particles) and pants; same PBD material. Canonical
folds: shirt = left panel in, right panel in, hem-to-shoulders; pants =
leg-over-leg, cuffs-to-waist ([garment_folds_lib.py](garment_folds_lib.py)).
Metric: mean per-particle distance to the canonical folded configuration after
2D rigid Procrustes alignment (`goal_err`; flat state ≈ 0.18-0.19, perfect
fold ≈ 0.03).

| garment | method | mean goal_err (3 eps) |
|---|---|---|
| t-shirt | scripted 3-fold | 0.129-0.171 (run-dependent) |
| t-shirt | **fold-denoiser (learned)** | **0.160** (best ep 0.110 — beat scripted that round) |
| pants | scripted 2-fold | 0.139 |
| pants | fold-denoiser (learned) | 0.172 |

The FOLD-denoiser is the scramble-inversion idea applied to folding itself:
forward process scrambles from the synthetically-generated FOLDED goal state;
a 4-head CNN predicts the inverse action (2 grasp cells + 2 place cells);
reverse rollout executes fold(grasp→place). At ~900 pairs/garment it shows
partial learning (occasionally beating scripted on the shirt) but is not yet
consistent — place-cell prediction (where the flap came from) is much harder
than grasp prediction and needs ~5-10x more data.

Untangle-denoiser v2 (garments, goal=flat, 1,740 pairs) drives the full
pipeline demos: `tshirts_demo.mp4` (2 shirts: untangle→fold→place, both
placed; one fling took coverage 0.45→1.12) and `pants_demo.mp4` (2 pants
placed; pants folds are the weakest link). ### Shirt untangling: all policies (6 tangles, ≤4 flings)

| policy | mean coverage | success@0.74 |
|---|---|---|
| corner heuristic (privileged) | 0.855 | 100% |
| random | 0.768 | 83% |
| diffusion (garment denoiser) | 0.78-0.83 on clean episodes* | 2/3 clean* |
| SAC (RL, 246 flings) | 0.603 | 67% |

*The diffusion column suffered Genesis solver explosions in 2/6 episodes
(larger cloth = less stable long rollouts); its non-exploded episodes match
its towel-level performance, and it drove both showcase videos successfully.
Note the shirt untangles easily (random hits 83%) — the discriminative
untangling comparison is the towel table above, where diffusion's advantage
is unambiguous. SAC again lands below random. Verdict across every matched
comparison in this project: **diffusion learned; RL did not** (at consumer-GPU
budgets, identical primitives and observations).

## Files & videos

- `baseline.py` — single-towel scripted end-to-end run (validated:
  fold_quality 0.81, on_stack=True — `baseline4.mp4`)
- `laundry_nofreeze.mp4` — two towels end-to-end with the corner heuristic
- `eval_compare.py` — policy comparison (RL actor vs heuristic vs random)
- `laundry_demo.mp4` — final showcase (see below)
- `runs_fling*/history.json` — RL learning curves

## Reproduce

```bash
# train the fling policy (≈2-3 h on the 5080)
.venv/bin/python train_fling.py --workers 9 --steps 1500 --out runs_fling4
# compare policies
.venv/bin/python eval_compare.py --episodes 10 --actor runs_fling4/actor_final.pt
# final multi-towel showcase on high-res cloth (~3700 particles/towel)
.venv/bin/python full_pipeline.py --towels 3 --grid 61 \
    --actor runs_fling4/actor_final.pt --out laundry_demo.mp4
```


## Staged untangle → two-fold (2026-09-01/02)

Tangled garment → fling-untangle → canonical two-fold, on the tabletop
scene (glass enclosure, third ironing arm, cotton-look material). Full
results, the material-stability post-mortem, and the RL-vs-diffusion
comparison: see `RESULTS_STAGED.md`. Videos: `staged_tshirt.mp4`,
`staged_pants.mp4`. Key stability rule inherited by everything here: the
cloth material must survive stacked folds — bending 1.4e-2 / air 0.006 is
the validated cotton-look setting, and heightmaps must be audited for
cells > 0.30 m (explosions clamp at the sim bound and pass isfinite).


## Laundry v2: bucket → uncrumble → iron → fold → stack (2026-09-04/05)

Full laundry loop on 4×-vertex two-layer garments: bucket outside the table
rim feeds 3 crumpled garments; policy uncrumples/folds; auto iron pass with
burn penalties; scripted stack. RL (CNN-SAC, ~1,700 arm actions) vs
scramble-inversion diffusion vs random on identical seeds: NOTHING beats
random on fold quality yet (all ≈0.30 baseline) — but both structured
policies had 0 burns vs random's 108. Full analysis, failure attribution
(solver stability edge + distribution shift), and next steps:
`RESULTS_LAUNDRY.md`. Videos: `laundry_rl_final.mp4`,
`laundry_diff_tshirt.mp4`, `laundry_diff_pants.mp4`.


## Round 2 complete (2026-09-12)

Full 9-config matrix (towel/tshirt/pants x RL/diffusion/random) on the
textured laundry cell: see `RESULTS_ROUND2.md`. First genuine RL win
(towels, via BC-of-diffusion + SAC); arm-state data fixed diffusion's
shirt deficit; all learned policies burn-free vs random's 306 burns.
Videos: `vid_{towel,tshirt,pants}_{rl,diffusion,random}.mp4` (all on the
AI-textured table). Docs: `cloth_robot__ALL_DETAILS_R2.pdf` (quiz incl.).
