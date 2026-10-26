"""Laundry-v2 RL environment: 3 crumpled garments in the bucket; per garment
the scripted feeder picks it onto the table, then the POLICY uncrumples,
irons (scripted pass triggered once flat), folds, and the scripted stacker
moves the result to the stack. Episode = all 3 garments.

Non-saturating fold metric (v1's Procrustes saturated near "flat"):
  fold_quality in [0,1] combines
    - area ratio: folded footprint / flat footprint, target ~0.27 (two-fold)
    - rectangularity: footprint fill of its bounding box
    - thinness: max height <= ~8 cloth thicknesses
Reward: phase-shaped (coverage gain while uncrumpling, fold_quality gain
while folding), iron burn penalty, per-action cost, stack bonus.

Obs (HM=32): heightmap (32x32, z-relative, over the table workspace)
  + scalars [coverage, area_ratio, rect, phase, garment_idx/2, burn_frac]
Action 8-d in [-1,1]: two grasp xy + two place xy (fling ignores places).
"""
import numpy as np
import gymnasium as gym

HM = 32
WS = 0.38
COV_OK = 0.70            # garments; towels use a stricter flatness gate
COV_OK_TOWEL = 0.80
N_GARMENTS = 3
ACTIONS_PER_GARMENT = 8
AREA_TARGET = 0.27


class LaundryEnv(gym.Env):
    metadata = {}

    def __init__(self, seed=0, garment="tshirt4.obj"):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (HM * HM + 6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
        self._seed0 = seed
        self._garment = garment
        self._built = False
        self._rng = np.random.default_rng(seed)

    # ---------- build ----------
    def _lazy_build(self):
        import scene as base
        base.PARTICLE_SIZE = (0.008 if "pants" in self._garment
                              else 0.0072)   # per-garment stress-tested
        # (towel4 runs at 0.0072 pending its stress verdict)
        base.SUBSTEPS = 15
        from scene_laundry import build_laundry_scene
        from primitives_laundry import LaundryDuo
        from primitives import Ironer
        from scene import settle, hold_home, HOME_L, HOME_R, HOME_I, TABLE_H
        garments = [self._garment] * N_GARMENTS
        scene, left, right, ironer, cloths, cam = build_laundry_scene(
            garments, seed=self._seed0)
        scene.build()
        left.set_dofs_position(HOME_L)
        right.set_dofs_position(HOME_R)
        ironer.set_dofs_position(HOME_I)
        left.control_dofs_position(HOME_L)
        right.control_dofs_position(HOME_R)
        ironer.control_dofs_position(HOME_I)
        settle(scene, 150)
        from metrics import particles
        self._flats = [particles(c).copy() for c in cloths]
        self._flat_area = self._footprint_area(self._flats[0])
        self.scene, self.cloths, self.cam = scene, cloths, cam
        self.duo = LaundryDuo(scene, left, right, cloths[0],
                              flat_pos=self._flats[0])
        self.ironer = Ironer(self.duo, scene)
        self._TH = TABLE_H
        self._built = True

    # ---------- metrics ----------
    @staticmethod
    def _footprint_area(pos, cell=0.02):
        ij = np.floor(pos[:, :2] / cell).astype(int)
        return len({(int(a), int(b)) for a, b in ij}) * cell * cell

    def _measures(self, pos):
        from metrics import coverage
        cov = coverage(pos)
        area = self._footprint_area(pos)
        ratio = area / max(self._flat_area, 1e-6)
        lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
        bbox = max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-6)
        rect = min(1.0, area / bbox)
        height = float(pos[:, 2].max() - self._TH)
        thin = float(np.clip(1.5 - height / 0.10, 0.0, 1.0))
        # fold quality peaks when ratio hits the two-fold target, footprint is
        # rectangular, and the stack is thin
        ar = float(np.exp(-((ratio - AREA_TARGET) / 0.10) ** 2))
        fq = 0.5 * ar + 0.3 * rect + 0.2 * thin
        return cov, ratio, rect, fq

    def _obs(self):
        from metrics import particles
        pos = particles(self.duo.cloth)
        hm = np.zeros((HM, HM), dtype=np.float32)
        ij = np.clip(((pos[:, :2] + WS) / (2 * WS) * HM).astype(int), 0, HM - 1)
        for (i, j), z in zip(ij, pos[:, 2] - self._TH):
            if z > hm[i, j]:
                hm[i, j] = z
        cov, ratio, rect, fq = self._measures(pos)
        gate = COV_OK_TOWEL if "towel" in self._garment else COV_OK
        phase = 1.0 if cov > gate else 0.0
        burn = float(self.ironer.burn_count) / 8.0
        obs = np.concatenate([hm.flatten() / 0.06,
                              [cov, ratio, rect, phase,
                               self._gi / 2.0, burn]]).astype(np.float32)
        return obs, cov, ratio, rect, fq, phase

    def _to_xy(self, a2, pos):
        lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
        t = (np.asarray(a2) + 1) / 2
        return lo + t * (hi - lo)

    # ---------- garment flow ----------
    def _feed_next_garment(self):
        """Scripted: activate garment self._gi and pick it from the bucket."""
        self.duo.set_active_cloth(self.cloths[self._gi],
                                  flat_pos=self._flats[self._gi])
        ok = self.duo.pick_from_bucket()
        if not ok:
            ok = self.duo.pick_from_bucket()   # one retry — picks are cheap
        from metrics import particles
        self.ironer.reset_coverage(len(particles(self.duo.cloth)))
        self._ironed_flat = False
        self._acts_this_garment = 0
        return ok

    def _maybe_iron(self, cov):
        """Scripted ironing when the garment first reads flat — keeps passing
        until >=95% of particles have been under the plate ("no skipped
        parts"), capped at 3 passes."""
        gate = COV_OK_TOWEL if "towel" in self._garment else COV_OK
        if not self._ironed_flat and cov > gate + 0.05:
            try:
                for k in range(2):
                    self.ironer.iron_pass(passes=1)
                    if self.ironer.iron_coverage >= 0.95:
                        break
                    self.ironer.cool_down()   # keep moving, never loiter
            except Exception:
                pass
            self._ironed_flat = True
            return True
        return False

    def _finish_garment(self):
        from primitives_laundry import stack_fold_result
        from scene_laundry import SHELF_POS
        # folded linen goes to the back-right shelf; small alternating offset
        dst = (SHELF_POS[0] - 0.04 + 0.02 * (self._gi % 2),
               SHELF_POS[1] - 0.06)
        stacked = stack_fold_result(self.duo, dst=dst)
        self.duo.release_all()
        self.duo.home(seconds=0.5)
        return stacked

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        if not self._built:
            self._lazy_build()
        from scene_laundry import crumple_into_bucket
        from scene import settle
        if hasattr(self.ironer, "heat"):
            self.ironer.heat[:] = 0.0
            self.ironer.burned[:] = False
        try:
            self.duo.release_all()
            self.duo.home(seconds=0.5)
        except Exception:
            pass
        for i, c in enumerate(self.cloths):
            p = self._flats[i] + [0, 0, 0.02]
            c.set_particles_pos(p)
            c.set_particles_vel(np.zeros_like(p))
        settle(self.scene, 40)
        for i, c in enumerate(self.cloths):
            self.duo.set_active_cloth(c, flat_pos=self._flats[i])
            try:
                crumple_into_bucket(self.scene, c, self._rng,
                                    folds=int(self._rng.integers(2, 4)),
                                    settle_steps=150)
            except Exception:
                pass                     # garment stays wherever it is
        self._gi = 0
        self._stacked = 0
        try:
            self._feed_next_garment()
        except Exception:
            self._ironed_flat = False
            self._acts_this_garment = 0
        obs, cov, ratio, rect, fq, phase = self._obs()
        self._fq, self._cov = fq, cov
        return obs, {"coverage": cov, "fold_q": fq}

    def step(self, action):
        from metrics import particles
        a = np.asarray(action, dtype=np.float64)
        obs0, cov0, r0, rect0, fq0, phase0 = self._obs()
        reward = -0.15
        crashed = False
        try:
            pos = particles(self.duo.cloth)
            if phase0 < 0.5:
                g1 = self._to_xy(a[0:2], pos)
                g2 = self._to_xy(a[2:4], pos)
                i = int(np.argmin(((pos[:, :2] - g1) ** 2).sum(1)))
                j = int(np.argmin(((pos[:, :2] - g2) ** 2).sum(1)))
                if i == j:
                    j = int(np.argmax(((pos[:, :2] - pos[i, :2]) ** 2).sum(1)))
                if pos[i, 0] > pos[j, 0]:
                    i, j = j, i
                self.duo.fling(None, None, particles_lr=(i, j))
            else:
                g1, g2 = self._to_xy(a[0:2], pos), self._to_xy(a[2:4], pos)
                q1, q2 = self._to_xy(a[4:6], pos), self._to_xy(a[6:8], pos)
                if g1[0] > g2[0]:
                    g1, g2, q1, q2 = g2, g1, q2, q1
                self.duo.fold([*g1], [*g2], [*q1], [*q2])
        except Exception:
            crashed = True
        self._acts_this_garment += 1

        obs, cov, ratio, rect, fq, phase = self._obs()
        burns0 = self.ironer.burn_count
        if self._maybe_iron(cov):
            # reward scales with iron COVERAGE (skipped parts cost), burns hurt
            reward += 1.0 * self.ironer.iron_coverage
            reward -= 1.5 * (self.ironer.burn_count - burns0)
            obs, cov, ratio, rect, fq, phase = self._obs()

        if phase0 < 0.5:
            reward += 8.0 * (cov - cov0)
        else:
            reward += 6.0 * (fq - fq0)

        done_garment = (fq > 0.75 and phase0 >= 0.5) or crashed \
            or self._acts_this_garment >= ACTIONS_PER_GARMENT
        if crashed:
            reward -= 2.0
        terminated = False
        if done_garment:
            # cleanup/transition steps can hit solver NaNs too — never let
            # them kill the worker (scene rebuild costs minutes); degrade to
            # an in-episode penalty instead
            try:
                if fq > 0.75 and not crashed:
                    reward += 4.0
                    if self._finish_garment():
                        reward += 2.0
                        self._stacked += 1
                else:
                    self.duo.release_all()
                    self.duo.home(seconds=0.5)
            except Exception:
                reward -= 1.0
            self._gi += 1
            if self._gi >= N_GARMENTS:
                terminated = True
                reward += 3.0 * self._stacked
            else:
                try:
                    if not self._feed_next_garment():
                        reward -= 1.0
                    obs, cov, ratio, rect, fq, phase = self._obs()
                except Exception:
                    reward -= 1.0
                    terminated = True   # scene too corrupted to continue
        self._fq, self._cov = fq, cov
        return obs, float(reward), terminated, False, \
            {"coverage": cov, "fold_q": fq, "stacked": self._stacked,
             "garment": self._gi, "crashed": crashed,
             "iron_cov": float(self.ironer.iron_coverage)}
