"""Staged RL environment: tangled garment -> UNTANGLE (flings) -> FOLD to the
canonical two-fold. Refined reward per user's directives:

- untangle phase: coverage gain, plus ORACLE SHAPING — the tangle generator
  knows the true inverse un-fold grasps (the user's hack), and actions near
  them earn a dense bonus on the first fling
- fold phase: goal-error reduction toward the canonical two-fold state
- success bonus when goal_err < GOAL_OK; small per-action cost

Obs: 16x16 heightmap + coverage + goal_err + phase (259-d).
Action: 8-d in [-1,1] — untangle uses [g1x,g1y,g2x,g2y]; fold uses all 8
        (two grasps + two placements).
"""
import numpy as np
import gymnasium as gym

HM = 16
WS = 0.35
COV_OK = 0.72
GOAL_OK = 0.06
MAX_ACTIONS = 8


class StagedEnv(gym.Env):
    metadata = {}

    def __init__(self, seed=0, garment="tshirt.obj"):
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                (HM * HM + 3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
        self._seed0 = seed
        self._garment = garment
        self._built = False
        self._rng = np.random.default_rng(seed)

    def _lazy_build(self):
        from scene import build_scene, settle, hold_home, HOME_L, HOME_R, TABLE_H
        from primitives import Duo
        from metrics import particles
        from garment_folds_lib import canonical_seq_for, reflect_fold
        scene, left, right, cloth, cam = build_scene(garment=self._garment,
                                                     seed=self._seed0)
        scene.build()
        left.set_dofs_position(HOME_L)
        right.set_dofs_position(HOME_R)
        hold_home(left, right)
        settle(scene, 100)
        self._flat = particles(cloth).copy()
        goal = self._flat.copy()
        for pt, n2 in canonical_seq_for(self._garment, self._flat):
            d = (goal[:, :2] - pt) @ n2
            goal = reflect_fold(goal, pt, n2, d > 0)
        self._goal = goal
        self.scene, self.cloth = scene, cloth
        self.duo = Duo(scene, left, right, cloth, flat_pos=self._flat)
        self._TH = TABLE_H
        self._built = True

    # ---------- helpers ----------
    def _goal_err(self, pos):
        a = pos[:, :2] - pos[:, :2].mean(0)
        b = self._goal[:, :2] - self._goal[:, :2].mean(0)
        u, s, vt = np.linalg.svd(a.T @ b)
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt
        return float(np.linalg.norm(a @ R - b, axis=1).mean())

    def _obs(self):
        from metrics import particles, coverage
        p = particles(self.cloth)
        hm = np.zeros((HM, HM), dtype=np.float32)
        ij = np.clip(((p[:, :2] + WS) / (2 * WS) * HM).astype(int), 0, HM - 1)
        for (i, j), z in zip(ij, p[:, 2] - self._TH):
            hm[i, j] = max(hm[i, j], z)
        cov = coverage(p)
        ge = self._goal_err(p)
        phase = 1.0 if cov > COV_OK else 0.0
        return (np.concatenate([hm.flatten() / 0.08, [cov, ge, phase]])
                .astype(np.float32), cov, ge, phase)

    def _to_xy(self, a2):
        from metrics import particles
        p = particles(self.cloth)
        lo, hi = p[:, :2].min(0), p[:, :2].max(0)
        t = (np.asarray(a2) + 1) / 2
        return lo + t * (hi - lo)

    def _nearest_particle(self, xy):
        from metrics import particles
        p = particles(self.cloth)
        return int(np.argmin(((p[:, :2] - xy) ** 2).sum(1)))

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        if not self._built:
            self._lazy_build()
        from scene import spawn_tangled, settle
        from metrics import particles
        self.duo.release_all()
        self.duo.home(seconds=0.4)
        self.cloth.set_particles_pos(self._flat + [0, 0, 0.02])
        self.cloth.set_particles_vel(np.zeros_like(self._flat))
        settle(self.scene, 60)
        self._oracle = spawn_tangled(self.scene, self.cloth, self._rng,
                                     settle_steps=200, return_oracle=True)
        self.t = 0
        self._first_fling = True
        obs, cov, ge, ph = self._obs()
        self._cov, self._ge = cov, ge
        return obs, {"coverage": cov, "goal_err": ge}

    def step(self, action):
        from metrics import particles
        a = np.asarray(action, dtype=np.float64)
        obs0, cov0, ge0, phase0 = self._obs()
        reward = -0.3
        try:
            if phase0 < 0.5:
                # UNTANGLE: fling with grasps from a[:4]
                g1 = self._to_xy(a[0:2])
                g2 = self._to_xy(a[2:4])
                i, j = self._nearest_particle(g1), self._nearest_particle(g2)
                if self._first_fling and self._oracle is not None:
                    p = particles(self.cloth)
                    o1, o2 = self._oracle
                    d = min(np.linalg.norm(p[i, :2] - p[o1, :2]) +
                            np.linalg.norm(p[j, :2] - p[o2, :2]),
                            np.linalg.norm(p[i, :2] - p[o2, :2]) +
                            np.linalg.norm(p[j, :2] - p[o1, :2])) / 2
                    reward += 2.0 * float(np.exp(-d / 0.08))   # oracle shaping
                self._first_fling = False
                p = particles(self.cloth)
                if p[i, 0] > p[j, 0]:
                    i, j = j, i
                self.duo.fling(None, None, particles_lr=(i, j))
            else:
                # FOLD: two grasps + two placements
                g1, g2 = self._to_xy(a[0:2]), self._to_xy(a[2:4])
                q1, q2 = self._to_xy(a[4:6]), self._to_xy(a[6:8])
                if g1[0] > g2[0]:
                    g1, g2, q1, q2 = g2, g1, q2, q1
                self.duo.fold([*g1], [*g2], [*q1], [*q2])
        except Exception:
            obs, cov, ge, ph = self._obs()
            return obs, -3.0, True, False, {"coverage": cov, "goal_err": ge,
                                            "crashed": True}
        self.t += 1
        obs, cov, ge, ph = self._obs()
        if phase0 < 0.5:
            reward += 10.0 * (cov - cov0)
        else:
            reward += 15.0 * (ge0 - ge)
        terminated = bool(ge < GOAL_OK and cov > 0.4)
        if terminated:
            reward += 8.0
        truncated = self.t >= MAX_ACTIONS
        self._cov, self._ge = cov, ge
        return obs, float(reward), terminated, truncated, \
            {"coverage": cov, "goal_err": ge}
