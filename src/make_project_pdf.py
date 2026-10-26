"""Generate cloth_robot__ALL_DETAILS_R<N>.pdf — the exhaustive project report.

Versioning: pass --rev N (default 1). Revisions NEVER overwrite older ones —
each revision is a new file (R1, R2, ...), matching the user's convention.
"""
import argparse
import datetime
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                PageBreak, Table, TableStyle)

ap = argparse.ArgumentParser()
ap.add_argument("--rev", type=int, default=1)
args = ap.parse_args()
OUT = f"cloth_robot__ALL_DETAILS_R{args.rev}.pdf"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists — bump --rev instead of overwriting")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=18)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=10,
                      leading=13.5, spaceAfter=6)
MONO = ParagraphStyle("MONO", parent=styles["Code"], fontSize=8.5, leading=11)


def T(rows, widths=None):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f5")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


S = []
P = lambda txt, st=BODY: S.append(Paragraph(txt, st))
SP = lambda h=8: S.append(Spacer(1, h))
PB = lambda: S.append(PageBreak())

# ---------------- title ----------------
S.append(Spacer(1, 1.6 * inch))
P("Cloth-Folding Robot Platform", ParagraphStyle(
    "T", parent=styles["Title"], fontSize=26))
P("Complete Project Documentation — Every Detail", ParagraphStyle(
    "ST", parent=styles["Title"], fontSize=14,
    textColor=colors.HexColor("#555555")))
SP(20)
P(f"Revision R{args.rev} &nbsp;•&nbsp; "
  f"{datetime.date.today().isoformat()} &nbsp;•&nbsp; "
  "Project: /c/q/RLE_cloth_robot_2026/robot_platform_experiments/"
  "08_cloth_folding", ParagraphStyle("meta", parent=BODY,
                                     alignment=1, fontSize=10))
P("Prepared for Aamer Khani. Older revisions are never deleted; each "
  "revision is a new file (R1, R2, ...).",
  ParagraphStyle("meta2", parent=BODY, alignment=1, fontSize=9,
                 textColor=colors.grey))
PB()

# ---------------- 1 overview ----------------
P("1. Project Overview and Goals", H1)
P("The project builds a multi-arm robotic laundry cell in the Genesis "
  "physics simulator and uses it as a rigorous testbed for a central "
  "scientific question: <b>can the user's scramble-inversion discrete "
  "diffusion method (Zenodo 21970238, 22063663) outperform reinforcement "
  "learning on real deformable-object manipulation, and at what cost?</b> "
  "The task ladder grew from towels on a floor to a full laundry loop: "
  "crumpled garments in a bucket hanging off a work table are picked one "
  "at a time by robot arms, untangled and flattened by two-arm flings, "
  "ironed by a third arm with full-coverage guarantees and burn physics, "
  "folded into a neat two-fold, and placed on a linen shelf.")
P("Three policy classes compete on identical episode seeds: (1) "
  "<b>Diffusion</b> — CNN denoisers trained on armless synthetic scramble "
  "data, executed as a reverse-process rollout; (2) <b>RL</b> — soft "
  "actor-critic over manipulation primitives, upgraded to a CNN encoder "
  "and behavior-cloning warm starts; (3) <b>Random</b> — the control that "
  "keeps everyone honest. Verdicts so far: diffusion decisively won "
  "towels round 1 (0.76 vs 0.42 coverage at ~50x lower cost), tied a "
  "privileged oracle on staged two-folds (metric saturation), and both "
  "learned methods tied random on the full laundry loop round 1 — a "
  "result traced to solver physics limits and training-data distribution "
  "shift, both addressed in the ongoing round 2.")

P("2. Hardware and Software Stack", H1)
S.append(T([
    ["Component", "Detail"],
    ["Simulator", "Genesis 1.3.3, GPU/Warp backend, PBD cloth solver"],
    ["GPU", "RTX 5080 laptop-class, 16 GB VRAM"],
    ["Robots", "3x Franka Emika Panda (MJCF), tabletop-mounted"],
    ["RL stack", "PyTorch SAC (MLP then CNN), parallel worker processes"],
    ["Diffusion stack", "4-conv CNN denoisers, 4 heads (2 grasp + 2 place "
     "cells over a 16x16 grid), soft-target CE loss"],
    ["Env API", "Gymnasium; 32x32 heightmap + 6 scalars obs; 8-d action"],
    ["Version control", "git repo rooted at /c/q; every change committed "
     "scoped to the project directory"],
], [1.5 * inch, 5.2 * inch]))

P("3. Scene Evolution", H1)
P("<b>v0 (towels era):</b> two floor-mounted Frankas facing a 0.4 m PBD "
  "towel; tangles spawned by teleported reflection folds; fold/stack zones "
  "on the floor.")
P("<b>v1 (garment era):</b> procedural t-shirt and pants meshes; feedback "
  "rounds fixed arm facing (both arms now face each other), added a large "
  "2.2 x 1.6 m work table at 0.75 m height with translucent glass "
  "enclosure walls, and a third arm holding an iron plate.")
P("<b>v2 (laundry era):</b> a bucket (0.42 m inner, 0.30 m deep) hangs "
  "entirely OUTSIDE the left table rim on a mounting bracket, aligned "
  "with the left arm and reachable through a 0.55 m gap in the left glass "
  "wall; three colored garments crumple-and-drop into it per episode; the "
  "ironing arm sits fully on the tabletop; a low linen shelf with a "
  "retaining lip occupies the back-right corner as the destination for "
  "folded output. PBD bounds extend below the tabletop to contain the "
  "bucket volume.")

P("4. Garment Assets", H1)
S.append(T([
    ["Mesh", "Verts", "Sim particles", "particle_size", "Notes"],
    ["towel_31 (v0)", "961", "~1,000", "0.0133", "round-1 towel"],
    ["tshirt.obj", "439", "~1,500", "0.0133", "single-layer, alphabet-T"],
    ["pants.obj", "426", "~1,400", "0.0133", "single-layer"],
    ["tshirt2/pants2", "878/852", "~2,900", "0.0133",
     "two-layer (front+back panels, seam walls, open hem)"],
    ["tshirt4.obj", "3,330", "6,592", "0.0072", "4x two-layer (current)"],
    ["pants4.obj", "3,306", "4,357", "0.008", "4x two-layer (current)"],
    ["towel4.obj", "3,721", "~6,900", "0.0072", "4x towel (current)"],
], [1.1 * inch, 0.7 * inch, 1.0 * inch, 0.9 * inch, 3.0 * inch]))
P("Two-layer construction: front and back panels stitched along boundary "
  "seams with the hem left open so the garment reads as real clothing. "
  "The PBD sampler resamples meshes into particles by particle_size, so "
  "mesh resolution and simulation resolution are tuned jointly.")
PB()

P("5. Physics Stability — the Defining Engineering Battle", H1)
P("<b>5.1 The material dilemma.</b> Cotton-look stiffness "
  "(bending_compliance 8e-3, air_resistance 0.02) explodes the solver "
  "whenever a fold lands on an already-folded stack. The failure is "
  "silent: exploded particles clamp at the PBD upper bound instead of "
  "going NaN, so isfinite() checks pass. A full day of folded-goal "
  "training data (100% of it) was garbage before a heightmap audit "
  "(any 16x16 cell &gt; 0.30 m = explosion) exposed it. The validated "
  "compromise material is bending 1.4e-2 / air 0.006, which survives the "
  "two-fold goal state plus three stacked scramble folds.")
P("<b>5.2 Resolution limits (stress-tested).</b> At 4x mesh resolution, "
  "particle_size 0.0072 passes for t-shirt (6,592 particles) and towel; "
  "pants need 0.008 (4,357). The 13k-particle regime (psize 0.0067) "
  "fails under every dt/substep/material combination tried — a hard "
  "boundary of this solver, documented for the planned custom engine.")
P("<b>5.3 Teleport fragility.</b> Teleporting a multi-fold tangle in one "
  "shot explodes even with stable material; the tangle spawner applies "
  "folds ONE at a time with damped settling between (velocity zeroing "
  "near surfaces + 2.5-3 m/s speed clamps). Repeated re-teleports to the "
  "folded goal state still explode 64-99% of episodes on 4x two-layer "
  "garments — the single biggest data-collection tax, and next-step #1 "
  "for the physics rewrite.")
P("<b>5.4 Hard-won primitive rules.</b> Never pin cloth inside gripper "
  "collision geometry (hands stop 11.5 cm above grasps); damp near-floor "
  "velocities every ~5 steps (PBD boundaries are frictionless); keep "
  "two-arm grasps &gt;12 cm apart; PBD bounds are load-bearing "
  "containment walls.")

P("6. Manipulation Primitives", H1)
S.append(T([
    ["Primitive", "What it does"],
    ["grasp", "descend, pinch TOP layer only (within one particle size of "
     "the highest candidate), pin to hand link; NaN guards; reach-clamped"],
    ["fling", "two-arm taut stretch, lift, forward fling, lay flat — the "
     "untangle workhorse"],
    ["fold", "two grasps + two placements over a hinge; lift-before-home"],
    ["carry", "two-arm transport of a folded item to a destination"],
    ["pick_from_bucket", "left arm reaches through the glass gap, below "
     "table level (special IK clamp zone), grasps the pile top, lifts "
     "over the rim, lays the garment mid-table; one retry on failure"],
    ["iron_pass", "third arm rasters the plate over the cloth bbox; row "
     "count derives from plate width with 25% overlap (no skipped "
     "stripes); repeat-until-95%-coverage with heat cool-down between "
     "passes"],
    ["stack/shelf", "carry the folded result to the back-right linen "
     "shelf with alternating offsets"],
], [1.3 * inch, 5.4 * inch]))

P("7. The Ironing Subsystem (full history)", H1)
P("The iron is a plate glued kinematically under the third arm's hand. "
  "Pressing flattens particles under the footprint and accumulates "
  "per-particle heat; heat beyond a threshold scorches (dark markers, "
  "negative reward). Full bug history, each found by validation and "
  "fixed with a scoped commit: (1) <b>rubber-band artifact</b> — the "
  "plate was re-pinned only on primitive steps, so direct scene.step "
  "paths let it free-fall then snap back; fixed by wrapping scene.step "
  "itself (plate-hand offset now 0.0 mm across all paths). (2) <b>fixed "
  "2-3 raster rows</b> left unpressed stripes on wide cloth; rows now "
  "tile the cloth width from plate geometry (iron coverage metric went "
  "0.72 -&gt; targeting 0.95+). (3) <b>repeat-until-covered scorching</b> "
  "— overlapping re-passes re-heated faster than per-step decay cooled "
  "(55 burns in one episode); fixed with 55% heat decay between passes. "
  "(4) <b>burn threshold retune</b> — BURN_HEAT 90 was calibrated for "
  "the old raster; correct full-coverage ironing legitimately reaches "
  "~120-150 heat-steps, so proper ironing burned (945 burns in one "
  "episode). Threshold now 260: only true loitering scorches. Iron "
  "coverage is a first-class eval metric.")
PB()

P("8. Metrics — Evolution Toward Honesty", H1)
S.append(T([
    ["Metric", "Status", "Lesson"],
    ["coverage (footprint/flat)", "active",
     "great for untangling; saturates once flat"],
    ["Procrustes goal_err", "retired for folding",
     "saturates near 'flat' (~0.15): even a privileged oracle cannot "
     "beat flat by much through real arms; produced a 3-way tie"],
    ["fold_q (area-ratio + rectangularity + thinness)", "active",
     "non-saturating proxy; peaks at the quarter-fold footprint"],
    ["FoldSSIM (height + gradient SSIM, principal-frame aligned, "
     "content-weighted)", "active",
     "sees SHAPE: perfect fold 1.0, settled fold 0.40, one-fold 0.21, "
     "crumpled blob 0.14, flat 0.04 — blobs no longer fool it"],
    ["iron_coverage", "active", "fraction of particles pressed; the "
     "'no skipped parts' guarantee made measurable"],
    ["burns", "active", "scorched-particle count; safety differentiator"],
], [2.4 * inch, 1.1 * inch, 3.2 * inch]))

P("9. The Diffusion Method (scramble-inversion)", H1)
P("<b>Forward process:</b> from a goal state (flat, or canonically "
  "folded), apply random reflection folds — cheap, armless, analytic. "
  "<b>Training:</b> a 4-conv CNN sees the scrambled heightmap and "
  "predicts the inverse action as four 16x16 cell distributions (grasp "
  "pair + place pair). Refined loss: Gaussian soft targets over cells, "
  "coverage-weighted samples, unordered-pair symmetry (min over head "
  "swap), fixed torch seeds with best-of-N model selection (training "
  "variance at few-hundred-sample scale is large). <b>Inference:</b> "
  "reverse rollout — repeatedly execute the predicted un-fold through "
  "arm primitives until coverage/fold criteria are met.")
P("<b>Distribution shift fix (round 2):</b> round-1 laundry denoisers "
  "trained on teleport-scrambles but evaluated on bucket-crumpled, "
  "arm-dropped states — and lost to random. The arm-state collector now "
  "generates flat-goal data through the REAL pipeline (crumple into "
  "bucket, arm pick, then scramble) and folded-goal data at the measured "
  "arm-drop position/orientation distribution. It also records "
  "env-native (observation, action) pairs for behavior cloning.")

P("10. The RL Method and Its Infrastructure", H1)
P("<b>Policy:</b> SAC; v1 MLP (65-d obs) grew to a CNN actor-critic "
  "(32x32 heightmap encoder, 256-d features, twin critics) per the "
  "'more complex model' directive. <b>Trainer evolution, each stage "
  "forced by a real failure:</b> parallel Genesis workers with "
  "crash-respawn -&gt; non-blocking respawn absorption (a fresh laundry "
  "worker needs ~10 min to build) -&gt; hang detection with 20-min "
  "poll-timeout kills (a worker once spun at 80% CPU for 3+ hours inside "
  "one solver step, silently stalling the whole fleet on a blocking pipe "
  "read) -&gt; replay-buffer persistence across restarts (three OS-level "
  "session restarts each cost the full buffer) -&gt; JSON-safe telemetry "
  "(a numpy float32 in history crashed a run at its first checkpoint). "
  "<b>Behavior cloning:</b> the CNN actor regresses the known-correct "
  "inverse actions; validation exposed multimodality (several valid "
  "grasp pairs per state), so BC serves as a weak warm start rather "
  "than a solution — an honest, documented limitation.")

P("11. Results Compendium (as of this revision)", H1)
P("<b>Towels round 1 (the flagship):</b> diffusion 0.76 coverage / 70% "
  "success vs privileged heuristic 0.42 / 30% vs SAC at random level — "
  "trained on 4,171 armless pairs in ~40 minutes, roughly 50x cheaper "
  "than the RL attempt it beat.")
P("<b>Shirts (flat, round 1):</b> diffusion clean episodes 0.78-0.83 "
  "coverage; SAC below random. Caveat: random flinging untangles shirts "
  "83% of the time, so shirts are a weak discriminator.")
P("<b>Staged untangle-to-two-fold:</b> three-way statistical tie at "
  "goal_err ~0.14 (diffusion 0.143, oracle 0.142, SAC 0.137-0.141) — "
  "the Procrustes metric saturates; fold-execution physics, not policy, "
  "is the bottleneck. Pants separated (oracle 0.115 vs diffusion 0.151) "
  "due to place-head data starvation.")
P("<b>Laundry loop round 1:</b> nothing beat random on fold quality "
  "(RL 0.328, random 0.334, diffusion 0.283-0.305); the one real "
  "difference was burns: structured policies 0, random 108. Causes "
  "diagnosed and addressed in round 2 (distribution shift; data "
  "starvation; RL cost).")
P("<b>Round 2 (in progress):</b> towels re-enter as the third garment "
  "class with 1,187 folded + 172 flat arm-state pairs; towel RL "
  "fine-tuned ~380 actions from a BC warm start; three-way towel evals "
  "running under the corrected iron physics at the time of writing. "
  "Shirts and pants follow, then nine showcase videos (3 garments x 3 "
  "policies) under the current scene.")
PB()

P("12. Cost Accounting", H1)
S.append(T([
    ["", "RL", "Diffusion"],
    ["training interactions", "~1,700 arm actions (laundry v1) + ~380 "
     "(towel v2)", "0 robot interactions"],
    ["wall-clock per dataset/run", "days, with babysitting",
     "~2 h collect + minutes to train"],
    ["cost per action/sample", "30-100 s (arm execution)",
     "~1-3 s (teleport scramble)"],
    ["infrastructure required", "respawn + hang-watchdog + replay "
     "persistence", "explosion filtering only"],
    ["scales with", "robot time (expensive)", "sim stability (fixable)"],
], [1.7 * inch, 2.5 * inch, 2.5 * inch]))

P("13. Engineering Practices", H1)
P("Autonomous long-horizon operation: background jobs with layered "
  "monitors (progress lines, failure signatures, liveness watchdogs); "
  "every stage validated on real outputs before being called done — a "
  "standing lesson from an early incident where a from-scratch training "
  "job produced a broken model that was reported as complete. Data "
  "audits (explosion heightmap rule) run inline in every collector. "
  "Every code change is committed immediately, scoped strictly to the "
  "project path (the git root /c/q is shared with unrelated projects); "
  "heavy artifacts (videos, checkpoints, datasets) are gitignored. "
  "Loud speaker announcements ('It's done') fire when full campaigns "
  "complete, timed to waking hours.")

P("14. Requirements for the Planned Custom Cloth Engine", H1)
P("Documented in CLOTH_ENGINE_REQUIREMENTS.md with a validation suite: "
  "(1) stability under 8 stacked layers at 5 mm spacing across a "
  "silk-to-denim stiffness sweep; (2) teleport-init tolerance (the "
  "enabler of cheap diffusion data); (3) plastic crease memory with "
  "temperature-dependent set rate (real ironing); (4) real boundary "
  "friction; (5) pinned particles as first-class contact citizens; "
  "(6) bounded per-step compute + deterministic replay; (7) 10x speed "
  "at ~6.6k particles; (8) explicit divergence flags — silent "
  "clamp-at-bounds explosions must be impossible.")

P("15. Artifact Inventory", H1)
S.append(T([
    ["Artifact", "Contents"],
    ["RESULTS_STAGED.md / RESULTS_LAUNDRY.md", "full per-round analyses"],
    ["CLOTH_ENGINE_REQUIREMENTS.md", "engine spec from measured pain"],
    ["40+ videos", "from debug clips to full laundry loops; 9-config "
     "showcase set in progress under the current scene"],
    ["Checkpoints", "SAC/CNN actors per era; 6 arm-state denoisers; "
     "BC actors"],
    ["Datasets", "data_scramble2/4/5, data_laundry_diff, data_armstate "
     "(explosion-filtered, BC-annotated)"],
    ["Envs/trainers", "env.py, env_staged.py, env_laundry.py; "
     "train_fling/staged/laundry/laundry2; bc_train.py"],
    ["Eval harnesses", "eval_compare, eval_staged, eval_laundry "
     "(actor/diffusion/random modes, dual metrics)"],
], [2.2 * inch, 4.5 * inch]))

P("16. Round-2 Final Results (added in R2)", H1)
P("Full 9-configuration matrix on identical seeds per garment — see "
  "RESULTS_ROUND2.md for the complete table. Headlines: the BC-diffusion-"
  "into-SAC hybrid scored the project's first genuine RL win (towels: "
  "fold_q 0.422 vs diffusion 0.395 vs random 0.372, 18/18 clean episodes); "
  "arm-state training data flipped diffusion's shirt deficit into a lead "
  "(0.330 vs random 0.309); all six learned configurations ran burn-free "
  "while random scorched 306 spots; every learned-policy solver crash came "
  "from the RL side; and the visual fold ceiling (best SSIM 0.064 vs ~0.40 "
  "for a settled ideal fold) still belongs to the physics engine, not the "
  "policies. Recommended production recipe: behavior-clone the diffusion "
  "policy, fine-tune with SAC.")

P("17. Project Quiz (answers overleaf)", H1)
P("Ten questions to test whether a reader truly absorbed this document — "
  "useful for onboarding collaborators.")
quiz = [
 "1. Why did 100% of an entire day's folded-goal dataset pass isfinite() "
 "checks while being complete garbage?",
 "2. What is the stable cotton-look material setting, and what happens at "
 "bending_compliance 8e-3?",
 "3. Why must tangle spawning apply folds one at a time instead of "
 "teleporting a multi-fold stack?",
 "4. The iron plate looked attached by a rubber band in early videos. "
 "What was the actual mechanism of that artifact?",
 "5. Why did the no-skips ironing loop initially scorch 55 burn spots in "
 "a single episode, and what two changes fixed it?",
 "6. Why was the Procrustes fold metric retired, and what saturates it?",
 "7. What distribution shift sank the round-1 laundry diffusion policy, "
 "and how did round 2 cure it?",
 "8. Why is behavior cloning of the scramble-inverse policy only a weak "
 "warm start rather than a full solution?",
 "9. What single line of evidence shows 'structure buys safety before it "
 "buys skill'?",
 "10. What is the highest-leverage next improvement, and why does it "
 "unlock the diffusion method specifically?",
]
for q in quiz:
    P(q)
PB()
P("Quiz Answers", H2)
answers = [
 "1. Exploded particles clamp at the PBD upper bound instead of going NaN "
 "— finite garbage. The audit rule: any 16x16 heightmap cell above 0.30 m "
 "marks an explosion.",
 "2. bending 1.4e-2 with air_resistance 0.006; at 8e-3 the solver explodes "
 "whenever any fold lands on an already-folded stack.",
 "3. A teleported multi-layer stack creates instant interpenetration "
 "impulses; sequential folds with damped settling (the collectors' recipe) "
 "stay stable.",
 "4. The plate was re-pinned to the hand only on primitive steps; direct "
 "scene.step paths (settle/damping loops) let it free-fall, then the next "
 "glued step teleported it back. Fix: wrap scene.step itself.",
 "5. Overlapping re-passes re-heated cloth faster than per-step decay "
 "cooled it. Fixes: 55% heat decay between passes, and raster rows derived "
 "from plate width so one pass truly covers (plus a burn threshold retuned "
 "so correct full coverage does not scorch).",
 "6. Even a privileged oracle executing perfect keypoints through real "
 "arms lands near the score of an untouched flat garment (~0.14-0.15) — "
 "rounded creases and placement slack saturate the metric.",
 "7. Denoisers trained on teleport-scramble states but evaluated on "
 "bucket-crumpled, arm-dropped states. Round 2 collected scrambles FROM "
 "arm-picked/drop-matched states.",
 "8. The action distribution is multimodal — several grasp pairs are "
 "equally valid per state — so MSE regression averages modes; BC "
 "validation error stays high (~0.75) even when the policy is useful.",
 "9. Burns: 0 across all six learned configurations vs 306 for random on "
 "towels (and 108 in round 1).",
 "10. A cloth engine stable for stacked folds at full resolution (the "
 "user's planned rewrite): diffusion's only real bottleneck is that "
 "folded-goal training states explode during collection — stability "
 "directly multiplies its cheap data supply.",
]
for a in answers:
    P(a)

P("18. Current Status and Roadmap", H1)
P("<b>Running now:</b> three-way towel evaluation under corrected iron "
  "physics. <b>Next:</b> towel showcase videos; shirt RL fine-tune "
  "(overnight) and evals; pants RL fine-tune and evals; nine "
  "configuration videos; the complete round-2 comparison report with "
  "fold_q + SSIM + iron coverage + burns on identical seeds; speaker "
  "announcement on completion. <b>Beyond:</b> the custom cloth engine "
  "(user's rewrite) unlocks full-resolution folded-goal data, which is "
  "the single highest-leverage improvement for the diffusion method; "
  "pixel-wise Q-maps remain the untried RL upgrade.")
SP(16)
P(f"— End of revision R{args.rev}. Next revision: R{args.rev + 1} "
  "(this file will not be modified).",
  ParagraphStyle("end", parent=BODY, textColor=colors.grey))

doc = SimpleDocTemplate(OUT, pagesize=letter,
                        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                        title=f"Cloth Robot — All Details R{args.rev}",
                        author="Aamer Khani / Claude")
doc.build(S)
print(f"wrote {OUT}")
