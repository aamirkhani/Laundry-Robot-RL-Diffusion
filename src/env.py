"""RL environment: learn WHERE to grasp a tangled towel for two-arm flings.

Observation: 8x8 max-height map over the workspace + coverage (65 floats).
Action: 4D in [-1,1] — two grasp points (ax1, ay1, ax2, ay2) mapped onto the
cloth's current bounding box and snapped to the nearest cloth particle.
Reward: coverage improvement; success bonus when the towel is flat enough
to fold (coverage > SUCCESS_COV). One episode = one tangled towel.
"""
import numpy as np
import gymnasium as gym

SUCCESS_COV = 0.74
MAX_ACTIONS = 6
GRID = 8
WS = 0.35   # heightmap half-extent (m)


class FlingEnv(gym.Env):
    metadata = {}

    def __init__(self, seed=0, record=False, garment=None):
        self._garment = garment
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (GRID * GRID + 1,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self._seed0 = seed
        self._record = record
        self._built = False
        self._rng = np.random.default_rng(seed)

    def _lazy_build(self):
        from scene import build_scene, settle, hold_home, HOME_L, HOME_R
        from primitives import Duo
        scene, left, right, cloth, cam = build_scene(n_grid=31, seed=self._seed0,
                                                     garment=self._garment)
        scene.build()
        left.set_dofs_position(HOME_L)
        right.set_dofs_position(HOME_R)
        hold_home(left, right)
        settle(scene, 120)
        from metrics import particles
        self._flat_pos = particles(cloth).copy()   # rest/flat particle layout
        self.scene, self.cloth = scene, cloth
        self.duo = Duo(scene, left, right, cloth, cam=cam, record=self._record,
                       flat_pos=self._flat_pos)
        self.cam = cam
        self._built = True

    # ---------- helpers ----------
    def _obs(self):
        from metrics import particles, coverage
        p = particles(self.cloth)
        hm = np.zeros((GRID, GRID), dtype=np.float32)
        ij = np.clip(((p[:, :2] + WS) / (2 * WS) * GRID).astype(int), 0, GRID - 1)
        from scene import TABLE_H
        for (i, j), z in zip(ij, p[:, 2] - TABLE_H):
            hm[i, j] = max(hm[i, j], z)
        cov = coverage(p)
        return np.concatenate([hm.flatten() / 0.08, [cov]]).astype(np.float32), cov

    def _action_to_points(self, a):
        from metrics import particles
        p = particles(self.cloth)
        lo, hi = p[:, :2].min(0), p[:, :2].max(0)
        pts = []
        for k in range(2):
            t = (a[2 * k:2 * k + 2] + 1) / 2
            raw = lo + t * (hi - lo)
            i = np.argmin(((p[:, :2] - raw) ** 2).sum(1))
            pts.append(p[i, :2].copy())
        return pts

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        if not self._built:
            self._lazy_build()
        from scene import spawn_tangled, settle
        self.duo.release_all()
        self.duo.home(seconds=0.4)
        # restore flat towel, zero velocities, re-tangle
        self.cloth.set_particles_pos(self._flat_pos + [0, 0, 0.02])
        self.cloth.set_particles_vel(np.zeros_like(self._flat_pos))
        settle(self.scene, 60)
        spawn_tangled(self.scene, self.cloth, self._rng, settle_steps=200)
        self.t = 0
        obs, cov = self._obs()
        self._cov = cov
        return obs, {"coverage": cov}

    def step(self, action):
        pts = self._action_to_points(np.asarray(action, dtype=np.float64))
        pl, pr = (pts[0], pts[1]) if pts[0][0] <= pts[1][0] else (pts[1], pts[0])
        self.duo.fling(pl, pr)
        self.t += 1
        obs, cov = self._obs()
        reward = 10.0 * (cov - self._cov) - 0.3
        terminated = bool(cov > SUCCESS_COV)
        if terminated:
            reward += 5.0
        truncated = self.t >= MAX_ACTIONS
        self._cov = cov
        return obs, float(reward), terminated, truncated, {"coverage": cov}
