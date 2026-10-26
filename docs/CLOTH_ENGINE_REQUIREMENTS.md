# Requirements for a cloth engine that supports real laundry manipulation

Distilled from the 08_cloth_folding campaign (towels → garments → staged
two-fold → full laundry loop) — every requirement below traces to a failure
we actually hit with Genesis 1.3.3 PBD on an RTX 5080.

## 1. Stability under stacked layers (the killer)

The two-fold goal state = 4 cloth layers. With bending stiff enough to look
like cotton (compliance ≤ 8e-3), ANY fold landing on that stack explodes the
solver. Even the tuned compromise (1.4e-2) explodes 64–99% of episodes on
4×-vertex two-layer garments at particle_size 0.0072–0.008. Concrete bar:

- 8 stacked layers, particle spacing 5 mm, bending stiffness spanning
  silk→denim, zero explosions over 10k randomized fold sequences.
- Explosions must be IMPOSSIBLE to mistake for success: clamp-at-bounds
  states passed isfinite() checks and silently poisoned 100% of a dataset.
  Prefer hard NaN or an explicit divergence flag queryable per step.

## 2. Teleport-init tolerance (what makes cheap data possible)

Scramble-inversion diffusion needs states INITIALIZED analytically (reflection
folds applied to particle arrays) rather than executed by arms — that is the
entire ~50× cost advantage. PBD tolerates one teleported fold with damped
settling but not multi-fold stacks. Requirement: any physically-plausible
layered configuration (layer gap ≥ cloth thickness) must settle stably from
a cold start, without hand-tuned clearances (we needed +0.04 m of air).

## 3. Plastic deformation / crease memory

PBD constraints are purely elastic: folds spring open, wrinkles have no
persistence, and "ironing" cannot exist except as our kinematic hack
(flatten particles under a plate footprint). Requirement: rest-state update
under sustained bending (crease sets after being held), with a
temperature-dependent rate so ironing = heat + pressure → faster set, and
scorching = a material state change, not painted-on markers.

## 4. Friction and boundary conditions

The PBD floor is frictionless — cloth slides forever; we damped velocities
manually every 5 steps in every script that touches cloth. Table/cloth and
cloth/cloth friction must hold a folded stack in place through adjacent arm
motion.

## 5. Pinning that coexists with contact

`fix_particles_to_link` inside gripper collision geometry → instant NaN.
We stopped hands 11.5 cm above every grasp as a workaround, which distorts
every manipulation primitive. Requirement: pinned particles are first-class —
they participate in contact resolution without energy pumping.

## 6. Determinism and hang-freedom

Solver steps occasionally spun forever at high particle counts (a worker at
80% CPU for 3+ h mid-step); we needed a 20-min kill-timeout in the trainer.
Requirement: bounded per-step compute, and bit-deterministic replay given a
seed (training infra and debugging both depend on it).

## 7. Performance targets (from measured pain)

- 6.6k-particle garment, 15 substeps, dt 5 ms: we got ~30-100 s per
  manipulation primitive (~10-20 s sim time). Target: 10× faster, because
  RL at 1,700 actions learned ~nothing and matching diffusion's data volume
  needs ~100k+ interactions.
- 3 garments (~20k particles) + 3 arms in one scene must fit interactive
  rates for the full laundry loop.

## 8. Fidelity floor

particle spacing 7.2 mm on a 0.44 m shirt was the stability limit; visible
fold creases and sleeve geometry want ≤ 4 mm. Two-layer construction
(front/back panels + seams, open hem) must drape without inter-layer
tunneling.

## Validation suite to ship with the engine

1. Stacked-fold stress: k=1..8 reflect folds + settle, assert finite/extent.
2. Teleport-init grid: random layered states, cold-start settle.
3. Crease persistence: fold, hold 5 s, release — measure spring-back angle.
4. Iron test: heat+pressure pass halves spring-back; dwell → scorch state.
5. Pin-grasp: pinned-corner drag at 1 m/s against table friction, no NaN.
6. Long-horizon soak: 1 h of random primitives, zero divergence, bounded
   step time.
