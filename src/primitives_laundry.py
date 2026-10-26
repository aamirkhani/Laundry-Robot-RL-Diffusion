"""Laundry-v2 primitives: Duo extended with multi-garment support and a
scripted pick-from-bucket (left arm reaches over the table rim, down into the
bucket, pins the top of the active garment, lifts it out and lays it on the
table center).
"""
import numpy as np

from primitives import Duo, Ironer, FINGER_OPEN
from scene import TABLE_H
import scene_laundry as SL


class LaundryDuo(Duo):
    def set_active_cloth(self, cloth, flat_pos=None):
        """Bind primitives (grasp/fling/fold/damp) to one of the garments."""
        self.release_all()
        self.cloth = cloth
        self.flat_pos = flat_pos

    def _clamp_reach(self, arm, pos):
        """Allow the LEFT arm below tabletop when working over the bucket."""
        p = np.asarray(pos, dtype=np.float64).copy()
        over_bucket = p[0] < -(SL.TABLE_SIZE[0] / 2) + 0.06
        if arm.name == "L" and over_bucket:
            base = np.array([-0.62, 0.0])
            v = p[:2] - base
            r = np.linalg.norm(v)
            r_new = np.clip(r, 0.28, 0.72)
            if r > 1e-6 and abs(r_new - r) > 1e-9:
                p[:2] = base + v / r * r_new
            p[2] = np.clip(p[2], SL.BUCKET_TOP - SL.BUCKET_D + 0.06,
                           TABLE_H + 0.62)
            return p
        return super()._clamp_reach(arm, pos)

    def pick_from_bucket(self, dst=(0.0, 0.0)):
        """Lift the ACTIVE garment out of the bucket onto the table at dst.
        Returns True if the garment ended up on the table."""
        from metrics import particles
        arm = self.L
        rim_wp = [-(SL.TABLE_SIZE[0] / 2) + 0.12, 0.0, TABLE_H + 0.35]
        self.move({arm: rim_wp}, seconds=0.8)
        pos = particles(self.cloth)
        if not np.isfinite(pos).all():
            return False
        top = int(np.argmax(pos[:, 2]))
        self.move({arm: [pos[top, 0], pos[top, 1], TABLE_H + 0.30]}, seconds=0.7)
        if not self.grasp(arm, particle=top, hover=0.05):
            self.move({arm: rim_wp}, seconds=0.6)
            self.home(seconds=0.6)
            return False
        # lift straight up and over the rim, then to the drop point
        pos = particles(self.cloth)
        g = pos[arm.held].mean(0)
        self.move({arm: [g[0], g[1], TABLE_H + 0.55]}, seconds=1.0)
        self.move({arm: rim_wp[:2] + [TABLE_H + 0.55]}, seconds=0.8)
        self.move({arm: [dst[0] - 0.05, dst[1], TABLE_H + 0.45]}, seconds=0.9)
        self.move({arm: [dst[0] - 0.05, dst[1], TABLE_H + 0.18]}, seconds=0.7)
        self.release(arm)
        self.settle_damped(50)
        self.home(seconds=0.7)
        self.settle_damped(40)
        pos = particles(self.cloth)
        on_table = (np.isfinite(pos).all()
                    and abs(float(np.median(pos[:, 0]))) < 0.5
                    and float(np.median(pos[:, 2])) > TABLE_H - 0.02)
        return bool(on_table)


def stack_fold_result(duo, dst=(0.0, 0.36)):
    """Carry the (folded) active garment to the stack zone."""
    from metrics import particles
    pos = particles(duo.cloth)
    if not np.isfinite(pos).all():
        return False
    c = pos[:, :2].mean(0)
    ext = pos[:, :2].max(0) - pos[:, :2].min(0)
    ax = int(np.argmax(ext))
    lo_i = int(np.argmin(pos[:, ax]))
    hi_i = int(np.argmax(pos[:, ax]))
    p1, p2 = pos[lo_i, :2], pos[hi_i, :2]
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    try:
        duo.carry(list(p1), list(p2), list(dst))
    except Exception:
        return False
    return True
