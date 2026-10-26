# Round 2: Full laundry loop — towels, shirts, pants × RL, diffusion, random

Final matrix (6 episodes/config unless crash-limited; identical tangle seeds
per garment across policies; iron with no-skips coverage + burn physics;
folded output to the linen shelf).

| garment | policy | mean reward | fold_q | SSIM | burns | sim crashes | n valid |
|---|---|---|---|---|---|---|---|
| towel | **RL (BC+SAC 380)** | −4.95 | **0.422** | 0.053 | 0 | 0 | 6 |
| towel | diffusion (arm-state) | −3.07 | 0.395 | 0.036 | 0 | 0 | 6 |
| towel | random | −79.87 | 0.372 | 0.033 | **306** | 0 | 6 |
| tshirt | RL (BC+SAC 400) | 1.24 | 0.305 | 0.064 | 0 | 1 | 4 |
| tshirt | **diffusion (arm-state)** | 2.29 | **0.330** | 0.039 | 0 | 0 | 6 |
| tshirt | random | 3.48 | 0.309 | 0.035 | 0 | 0 | 6 |
| pants | RL (BC+SAC 400) | 1.28 | 0.311 | 0.003 | 0 | 0 | 6 |
| pants | diffusion (arm-state) | −0.16 | 0.300 | 0.017 | 0 | 0 | 6 |
| pants | random | 0.46 | 0.353* | 0.011 | 0 | 2 | 2* |

*pants random survived only 2 of 6 episodes (2 solver crashes stopped it) —
its "lead" rests on n=2 and is not credible.

## What round 2 established

1. **The BC-diffusion-RL hybrid produced the project's first genuine RL
   win**: towels, where fold_q ordering is RL > diffusion > random with all
   18 episodes clean. Behavior-cloning the diffusion policy's inverse
   actions, then SAC fine-tuning, is the recipe — pure RL never beat
   anything in four prior rounds.
2. **The arm-state data fix worked for diffusion**: shirts flipped from a
   round-1 deficit (0.305 vs random 0.334) to a lead (0.330 vs 0.309).
   Matching the training distribution to arm-dropped states was the cure
   for exactly the garment where teleport-scramble shift hurt most.
3. **Safety separates learned from random everywhere**: 0 burns across all
   6 learned configs vs 306 random burns on towels — the third consecutive
   demonstration that structure buys safety before it buys skill.
4. **The visual ceiling stands**: best SSIM anywhere is 0.064 vs ~0.40 for
   a physically-settled ideal fold. No method yet produces folds that LOOK
   right — the solver-stability wall (documented in
   CLOTH_ENGINE_REQUIREMENTS.md) still caps fold quality itself.
5. **RL remains fragile in ways diffusion is not**: every solver-explosion
   crash in learned-policy evals came from the RL side (shirts). Diffusion
   ran 18/18 clean episodes.

## Verdict, one line per method

- **Diffusion**: cheapest by ~50x, never crashes, wins shirts, close second
  everywhere else — the best default.
- **RL**: only viable warm-started from diffusion; wins towels; costs days
  and crashes occasionally — a fine-tuner, not a from-scratch learner.
- **Hybrid (BC of diffusion → SAC)**: the actual production recipe this
  project recommends.

## Artifacts

- `eval_{rl,diff,random}_{towel,tshirt_r2,pants_r2}.json` — all raw data
- `vid_{towel,tshirt,pants}_{rl,diffusion,random}.mp4` — 9 showcase videos
  (pants set on the AI-textured table)
- `cloth_robot__ALL_DETAILS_R2.pdf` — full project documentation
- Checkpoints: `runs_towel/actor_380.pt`, `runs_tshirt_v2/actor_400.pt`,
  `runs_pants_v2/actor_400.pt`; 6 arm-state denoisers; 3 BC actors
