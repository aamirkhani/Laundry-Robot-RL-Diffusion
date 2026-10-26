"""Bimanual cloth-manipulation primitives for the two Frankas.

Grasping = pin the cloth particles nearest the gripper to the hand link
(fix_particles_to_link); release = release_particle. Arms move via IK with
joint-space interpolation. All primitives step the sim and optionally record.
"""
import numpy as np
import genesis as gs

from scene import TABLE_H

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # gripper pointing straight down
FINGER_OPEN = 0.04
GRASP_RADIUS = 0.025


class Arm:
    def __init__(self, entity, name):
        self.e = entity
        self.name = name
        self.hand = entity.get_link("hand")
        self.held = None          # np.array of grasped particle indices or None

    @property
    def hand_pos(self):
        p = self.e.get_links_pos()[self.hand.idx_local]
        return np.asarray(p.cpu() if hasattr(p, "cpu") else p)


class Duo:
    """Coordinates the two arms + cloth + scene stepping/recording.

    flat_pos: the towel's flat rest particle layout (same particle order as the
    live cloth) — gives cloth-space distances for taut stretching."""

    def __init__(self, scene, left, right, cloth, cam=None, record=False, every=4,
                 flat_pos=None):
        self.scene = scene
        self.cloth = cloth
        self.cam = cam
        self.record = record
        self.every = every
        self.flat_pos = flat_pos
        self.L = Arm(left, "L")
        self.R = Arm(right, "R")
        self._steps = 0

    # ---------- low level ----------
    def step(self, n=1):
        for _ in range(n):
            self.scene.step()
            if self.record and self.cam is not None and self._steps % self.every == 0:
                self.cam.render()
            self._steps += 1
            if self._steps % 5 == 0:
                self.damp_floor()

    MAX_VEL = 3.0   # m/s — PBD explosions launch particles far faster

    def damp_floor(self):
        """Stability guards: (1) the PBD floor boundary is frictionless — kill
        horizontal velocity of near-floor particles (proxy friction); (2) clamp
        particle speed globally — pinned-particle whips can pump energy into
        the solver until the cloth explodes."""
        pos = self.cloth.get_particles_pos().cpu().numpy()
        vel = self.cloth.get_particles_vel().cpu().numpy()
        if pos.ndim == 3:
            pos, vel = pos[0], vel[0]
        changed = False
        mask = pos[:, 2] < TABLE_H + 0.025
        if mask.any():
            vel[mask, 0] = 0.0
            vel[mask, 1] = 0.0
            changed = True
        speed = np.linalg.norm(vel, axis=1)
        fast = speed > self.MAX_VEL
        if fast.any():
            vel[fast] *= (self.MAX_VEL / speed[fast])[:, None]
            changed = True
        if changed:
            self.cloth.set_particles_vel(vel)

    def settle_damped(self, steps=40, chunk=10):
        for _ in range(max(1, steps // chunk)):
            self.step(chunk)
            self.damp_floor()

    def _clamp_reach(self, arm, pos):
        """Keep IK targets inside the arm's comfortable annulus."""
        base = np.array([-0.62, 0.0]) if arm.name == "L" else np.array([0.62, 0.0])
        p = np.asarray(pos, dtype=np.float64).copy()
        v = p[:2] - base
        r = np.linalg.norm(v)
        r_new = np.clip(r, 0.28, 0.70)
        if r > 1e-6 and abs(r_new - r) > 1e-9:
            p[:2] = base + v / r * r_new
        p[2] = np.clip(p[2], TABLE_H + 0.005, TABLE_H + 0.62)
        return p

    def _ik(self, arm, pos):
        pos = self._clamp_reach(arm, pos)
        q = arm.e.inverse_kinematics(link=arm.hand, pos=np.asarray(pos, dtype=np.float32),
                                     quat=DOWN_QUAT)
        return np.asarray(q.cpu() if hasattr(q, "cpu") else q)

    def move(self, targets, seconds=0.8, keep_fingers=FINGER_OPEN):
        """targets: dict {arm: xyz or None}. Joint-space interpolation, synced."""
        plans = {}
        for arm, pos in targets.items():
            if pos is None:
                continue
            cur = np.asarray(arm.e.get_dofs_position().cpu())
            goal = self._ik(arm, pos)
            goal[7:9] = keep_fingers if arm.held is None else 0.0
            plans[arm] = (cur, goal)
        n = max(2, int(seconds / self.scene.sim_options.dt))
        for i in range(n):
            a = (i + 1) / n
            s = 3 * a**2 - 2 * a**3       # smoothstep
            for arm, (cur, goal) in plans.items():
                arm.e.control_dofs_position(cur + s * (goal - cur))
            self.step()

    # ---------- grasp / release ----------
    def grasp(self, arm, xy=None, particle=None, hover=0.12):
        """Descend and pin cloth to the hand link. Target is either a particle
        index (re-resolved to its CURRENT position — robust to the pile being
        disturbed by the other arm) or an (x, y) location."""
        pos = self.cloth.get_particles_pos().cpu().numpy()
        if pos.ndim == 3:
            pos = pos[0]
        if particle is not None:
            target = pos[int(particle)]
            if not np.isfinite(target).all():
                return False                  # cloth state corrupted (NaN)
        else:
            d2 = ((pos[:, :2] - np.asarray(xy)) ** 2).sum(1)
            if not np.isfinite(d2).any():
                return False                  # cloth state corrupted (NaN)
            near = d2 < 0.03 ** 2
            if near.sum() == 0:
                near = d2 < 0.06 ** 2
            if near.sum() == 0:
                near = d2 <= np.nanmin(d2) + 1e-9   # nearest finite particle
            if near.sum() == 0:
                return False
            target = pos[near][np.argmax(pos[near][:, 2])]
        self.move({arm: [target[0], target[1], target[2] + hover]}, seconds=0.6)
        self.move({arm: [target[0], target[1], target[2] + 0.115]}, seconds=0.5)
        pos = self.cloth.get_particles_pos().cpu().numpy()
        if pos.ndim == 3:
            pos = pos[0]
        tip = arm.hand_pos + np.array([0, 0, -0.10])   # fingertip approx
        # abort if the reach-clamp kept us away from the cloth: pinning cloth at
        # a distance creates violent rope-drag dynamics
        if np.linalg.norm(pos[:, :2] - tip[:2], axis=1).min() > 0.06:
            return False
        d = np.linalg.norm(pos - tip, axis=1)
        cand = np.where(d < GRASP_RADIUS)[0]
        if len(cand) == 0:
            cand = np.array([np.argmin(d)])
        # pinch ONLY the top layer: keep candidates within one particle size of
        # the highest candidate (grabbing through layers clamps folds shut)
        z_top = pos[cand][:, 2].max()
        grabbed = cand[pos[cand][:, 2] > z_top - 0.012]
        self.cloth.fix_particles_to_link(arm.hand.idx, particles_idx_local=grabbed)
        arm.held = grabbed
        self.step(20)
        return True

    def release(self, arm):
        if arm.held is not None:
            self.cloth.release_particle(arm.held)
            arm.held = None
            self.step(30)

    def release_all(self):
        self.release(self.L)
        self.release(self.R)

    # ---------- primitives ----------
    def home(self, seconds=0.8):
        from scene import HOME_L, HOME_R
        curL = np.asarray(self.L.e.get_dofs_position().cpu())
        curR = np.asarray(self.R.e.get_dofs_position().cpu())
        n = max(2, int(seconds / self.scene.sim_options.dt))
        for i in range(n):
            a = (i + 1) / n
            s = 3 * a**2 - 2 * a**3
            self.L.e.control_dofs_position(curL + s * (HOME_L - curL))
            self.R.e.control_dofs_position(curR + s * (HOME_R - curR))
            self.step()

    def ensure_centered(self, max_r=0.25):
        """If the towel drifted out of the shared workspace, carry it back."""
        from metrics import particles, find_corners
        pos = particles(self.cloth)
        c = pos[:, :2].mean(0)
        if np.linalg.norm(c) <= max_r:
            return
        cn = find_corners(pos)
        g1, g2 = (cn[0], cn[3]) if cn[0][0] <= cn[3][0] else (cn[3], cn[0])
        self.carry(g1, g2, np.zeros(2), height=TABLE_H + 0.07)

    def cloth_space_dist(self):
        """Distance between the two arms' grasped regions in the FLAT layout."""
        if self.flat_pos is None or self.L.held is None or self.R.held is None:
            return None
        a = self.flat_pos[self.L.held][:, :2].mean(0)
        b = self.flat_pos[self.R.held][:, :2].mean(0)
        return float(np.linalg.norm(a - b))

    def fling(self, p1, p2, height=None, spread=0.42, forward=0.15,
              particles_lr=None):
        """Two-arm dynamic fling (FlingBot-style): grasp two points, lift high,
        carry to the workspace center, stretch apart along x, swing forward-down
        fast so the cloth unfurls flat — always landing centered.
        particles_lr: optional (idxL, idxR) particle indices to grasp instead
        of xy points (robust when the first grasp disturbs the pile)."""
        if height is None:
            height = TABLE_H + 0.55
        pos_all = self.cloth.get_particles_pos().cpu().numpy()
        if pos_all.ndim == 3:
            pos_all = pos_all[0]
        if particles_lr is not None:
            g1 = pos_all[int(particles_lr[0])][:2]
            g2 = pos_all[int(particles_lr[1])][:2]
        else:
            g1, g2 = np.asarray(p1)[:2], np.asarray(p2)[:2]
        # too-close grasps would drive the two grippers into each other
        if np.linalg.norm(g1 - g2) < 0.12:
            if particles_lr is not None:
                ok1 = self.grasp(self.L, particle=particles_lr[0])
            else:
                ok1 = self.grasp(self.L, p1)
            ok2 = False
        elif particles_lr is not None:
            ok1 = self.grasp(self.L, particle=particles_lr[0])
            ok2 = self.grasp(self.R, particle=particles_lr[1])
        else:
            ok1 = self.grasp(self.L, p1)
            ok2 = self.grasp(self.R, p2)
        if not (ok1 or ok2):
            return
        # 1) lift straight up at the grasp locations (no dragging on the floor)
        g1 = self.L.hand_pos
        g2 = self.R.hand_pos
        self.move({self.L: [g1[0], g1[1], height], self.R: [g2[0], g2[1], height]},
                  seconds=0.6)
        # 2) bring to a centered narrow hold, then STRETCH slowly until taut.
        #    Taut = the cloth-space distance between the grasped regions; pulling
        #    further than the cloth allows makes the arms buckle.
        y0 = -0.02
        csd = self.cloth_space_dist()
        if csd is not None:
            spread = float(np.clip(csd * 0.92, 0.14, spread))
        self.move({self.L: [-0.10, y0, height], self.R: [0.10, y0, height]}, seconds=0.6)
        self.move({self.L: [-spread / 2, y0, height], self.R: [spread / 2, y0, height]},
                  seconds=1.0)
        # 3) shake: two quick vertical snaps to drop remaining tangles
        for _ in range(2):
            self.move({self.L: [-spread / 2, y0, height - 0.06],
                       self.R: [spread / 2, y0, height - 0.06]}, seconds=0.18)
            self.move({self.L: [-spread / 2, y0, height],
                       self.R: [spread / 2, y0, height]}, seconds=0.18)
        # 4) lay the hanging curtain down deterministically: arc down-and-back
        #    so the cloth unrolls onto the floor in front of the grippers
        for frac, z in ((0.30, TABLE_H + 0.36), (0.65, TABLE_H + 0.16),
                        (1.0, TABLE_H + 0.02)):
            yb = y0 + 0.12 - 0.57 * frac
            self.move({self.L: [-spread / 2, yb, z], self.R: [spread / 2, yb, z]},
                      seconds=0.35)
        self.step(40)   # cloth settles taut on the floor while still held
        self.release_all()
        self.home()
        self.settle_damped(50)

    def carry(self, p1, p2, dst_center, height=None, place_z=None):
        """Two-arm carry: grasp p1/p2, lift a little, translate both grasps by
        the same delta so the folded towel arrives at dst_center intact."""
        if height is None:
            height = TABLE_H + 0.08
        if place_z is None:
            place_z = TABLE_H + 0.025
        ok1 = self.grasp(self.L, p1)
        ok2 = self.grasp(self.R, p2)
        if not (ok1 or ok2):
            return
        mid = (np.asarray(p1) + np.asarray(p2)) / 2
        delta = np.asarray(dst_center) - mid
        a = np.asarray(p1) + delta
        b = np.asarray(p2) + delta
        self.move({self.L: [p1[0], p1[1], height], self.R: [p2[0], p2[1], height]}, seconds=0.5)
        self.move({self.L: [a[0], a[1], height], self.R: [b[0], b[1], height]}, seconds=0.9)
        self.move({self.L: [a[0], a[1], place_z], self.R: [b[0], b[1], place_z]}, seconds=0.4)
        self.release_all()
        # lift straight up before homing so the arms never drag through the
        # placed towel (stack-plowing prevention)
        self.move({self.L: [a[0], a[1], TABLE_H + 0.30],
                   self.R: [b[0], b[1], TABLE_H + 0.30]}, seconds=0.4)
        self.home()
        self.settle_damped(50)

    def drag(self, arm, src, dst, height=None):
        """Grasp at src, slide low to dst, release (repositions cloth parts)."""
        if height is None:
            height = TABLE_H + 0.05
        if not self.grasp(arm, src):
            return
        self.move({arm: [src[0], src[1], height]}, seconds=0.4)
        self.move({arm: [dst[0], dst[1], height]}, seconds=0.8)
        self.move({arm: [dst[0], dst[1], TABLE_H + 0.02]}, seconds=0.3)
        self.release(arm)
        self.home()
        self.settle_damped(40)

    def fold(self, p1, p2, q1, q2, height=None):
        """Two-arm fold: grasp p1/p2 (an edge), arc up-and-over, lay on q1/q2."""
        if height is None:
            height = TABLE_H + 0.16
        ok1 = self.grasp(self.L, p1)
        ok2 = self.grasp(self.R, p2)
        if not (ok1 or ok2):
            return
        m1 = (np.asarray(p1) + np.asarray(q1)) / 2
        m2 = (np.asarray(p2) + np.asarray(q2)) / 2
        self.move({self.L: [p1[0], p1[1], height], self.R: [p2[0], p2[1], height]}, seconds=0.5)
        self.move({self.L: [m1[0], m1[1], height + 0.06],
                   self.R: [m2[0], m2[1], height + 0.06]}, seconds=0.6)
        self.move({self.L: [q1[0], q1[1], TABLE_H + 0.03],
                   self.R: [q2[0], q2[1], TABLE_H + 0.03]}, seconds=0.6)
        self.step(30)   # let the flap set before releasing
        self.release_all()
        # lift straight up before homing — dragging arms across the fresh fold
        # plows it open
        self.move({self.L: [q1[0], q1[1], TABLE_H + 0.30],
                   self.R: [q2[0], q2[1], TABLE_H + 0.30]}, seconds=0.4)
        self.home()
        self.settle_damped(60)



class Ironer:
    """Third arm with a flat iron plate. iron_pass() rasters the plate just
    above a cloth region; particles under the plate get pressed flat
    (z pulled to layer height, velocity damped) — the 'ironed' look."""

    IRON_HOVER = 0.125    # hand height above cloth top during a pass
    PLATE = (0.14, 0.09)
    BURN_HEAT = 260   # retuned for plate-width raster (7 rows + overlap + repeat): correct full-coverage ironing peaks ~120-150 heat-steps; only genuine loitering exceeds 260        # press-ticks of dwell before scorching (raster is safe)

    def __init__(self, duo, scene):
        self.duo = duo
        self.scene = scene
        self.e = scene._ironer
        self.plate = scene._iron_plate
        self.hand = self.e.get_link("hand")
        from scene import HOME_I
        self.home_q = HOME_I
        # glue the plate to the hand on EVERY sim step. Wrapping duo.step is
        # NOT enough: settle()/damped loops call scene.step() directly, and
        # during those steps the free plate falls under gravity, then snaps
        # back on the next glued step (the "rubber band" artifact). Wrap the
        # SCENE's step so no physics step escapes the glue.
        if not getattr(scene, "_iron_glued", False):
            orig_scene_step = scene.step

            def scene_step_glued(*a, **kw):
                out = orig_scene_step(*a, **kw)
                self._glue_plate()
                return out
            scene.step = scene_step_glued
            scene._iron_glued = True

    def _hand_pos(self):
        p = self.e.get_links_pos()[self.hand.idx_local]
        return np.asarray(p.cpu() if hasattr(p, "cpu") else p)

    def reset_coverage(self, n_particles):
        """Per-garment ironed mask — call when a new garment is activated."""
        self.ironed = np.zeros(n_particles, dtype=bool)

    @property
    def iron_coverage(self):
        m = getattr(self, "ironed", None)
        if m is None or len(m) == 0:
            return 0.0
        return float(m.mean())

    def _glue_plate(self):
        hp = self._hand_pos()
        self.plate.set_pos(np.array([hp[0], hp[1], hp[2] - 0.115]))
        try:
            self.plate.set_dofs_velocity(np.zeros(6))
        except Exception:
            pass
        # keep placed scorch markers pinned where the burns happened
        for ent, site in getattr(self, "_scorch_sites", []):
            ent.set_pos(site)
            try:
                ent.set_dofs_velocity(np.zeros(6))
            except Exception:
                pass

    def _move(self, pos, seconds=0.6):
        q = self.e.inverse_kinematics(link=self.hand,
                                      pos=np.asarray(pos, dtype=np.float32),
                                      quat=DOWN_QUAT)
        q = np.asarray(q.cpu() if hasattr(q, "cpu") else q)
        cur = np.asarray(self.e.get_dofs_position().cpu())
        n = max(2, int(seconds / self.scene.sim_options.dt))
        for i in range(n):
            a = (i + 1) / n
            sm = 3 * a**2 - 2 * a**3
            self.e.control_dofs_position(cur + sm * (q - cur))
            self.duo.step()
            self._glue_plate()
            if i % 3 == 0:
                self._press()

    def _press(self):
        """Flatten cloth under the plate footprint; accumulate heat — dwelling
        too long BURNS the cloth (scorch mark + negative-reward signal)."""
        cloth = self.duo.cloth
        pos = cloth.get_particles_pos().cpu().numpy()
        vel = cloth.get_particles_vel().cpu().numpy()
        if pos.ndim == 3:
            pos, vel = pos[0], vel[0]
        hp = self._hand_pos()
        px, py = self.PLATE[0] / 2, self.PLATE[1] / 2
        under = ((np.abs(pos[:, 0] - hp[0]) < px) &
                 (np.abs(pos[:, 1] - hp[1]) < py))
        if not under.any():
            return
        z = pos[under, 2]
        rel = z - TABLE_H
        # press toward crisp layers (2 particle-layers max under the iron)
        target = TABLE_H + np.clip(rel, 0.004, 0.03)
        pos2 = pos.copy()
        pos2[under, 2] = 0.4 * target + 0.6 * z
        vel[under] *= 0.2
        cloth.set_particles_pos(pos2)
        cloth.set_particles_vel(vel)
        # heat bookkeeping
        if not hasattr(self, "heat"):
            self.heat = np.zeros(len(pos))
            self.burned = np.zeros(len(pos), dtype=bool)
            self._scorch_used = 0
        self.heat[under] += 1.0
        if hasattr(self, "ironed") and len(self.ironed) == len(under):
            self.ironed |= under
        self.heat[~under] *= 0.97
        newly = (self.heat > self.BURN_HEAT) & (~self.burned)
        if newly.any():
            self.burned |= newly
            pool = getattr(self.scene, "_scorch_pool", [])
            if not hasattr(self, "_scorch_sites"):
                self._scorch_sites = []
            if self._scorch_used < len(pool):
                c = pos[newly].mean(0)
                site = np.array([c[0], c[1], max(c[2] + 0.004, TABLE_H + 0.008)])
                pool[self._scorch_used].set_pos(site)
                self._scorch_sites.append((pool[self._scorch_used], site))
                self._scorch_used += 1
            print(f"    [!] IRON BURN: {int(newly.sum())} particles scorched "
                  f"(total {int(self.burned.sum())})", flush=True)

    @property
    def burn_count(self):
        return int(getattr(self, "burned", np.zeros(1)).sum())

    def home(self, seconds=0.7):
        cur = np.asarray(self.e.get_dofs_position().cpu())
        n = max(2, int(seconds / self.scene.sim_options.dt))
        for i in range(n):
            a = (i + 1) / n
            sm = 3 * a**2 - 2 * a**3
            self.e.control_dofs_position(cur + sm * (self.home_q - cur))
            self.duo.step()
            self._glue_plate()

    def cool_down(self, factor=0.45):
        """Global heat decay between raster passes — overlapping rows re-heat
        faster than the per-step decay dissipates, so repeated coverage passes
        were scorching the cloth (55 burns in one eval episode)."""
        if hasattr(self, "heat"):
            self.heat *= factor

    def iron_pass(self, passes=2):
        """Raster the iron over the cloth's current bounding box. Row count
        derives from the PLATE's y-width with ~25% overlap — a fixed 2-3 rows
        left unpressed stripes on anything wider than the plate ("skipped
        parts"); `passes` now only scales extra repetitions."""
        from metrics import particles
        pos = particles(self.duo.cloth)
        lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
        lo = np.maximum(lo, [-0.42, -0.40])
        hi = np.minimum(hi, [0.42, 0.40])
        ztop = float(pos[:, 2].max())
        zpass = ztop + self.IRON_HOVER - 0.02
        span_y = max(hi[1] - lo[1], 0.05)
        n_rows = max(2, int(np.ceil(span_y / (self.PLATE[1] * 0.75))) + 1)
        ys = np.linspace(lo[1] + 0.03, hi[1] - 0.03, n_rows)
        self._move([lo[0], ys[0], zpass + 0.15], seconds=0.7)
        for k, y in enumerate(ys):
            x0, x1 = (lo[0], hi[0]) if k % 2 == 0 else (hi[0], lo[0])
            self._move([x0, y, zpass], seconds=0.35)
            self._move([x1, y, zpass], seconds=1.1)
        self._move([hi[0], ys[-1], zpass + 0.2], seconds=0.5)
        self.home()
        self.duo.settle_damped(30)
