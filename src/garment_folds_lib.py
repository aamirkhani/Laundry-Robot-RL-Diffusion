"""Canonical garment fold keypoints (shared by eval + pipeline)."""
import os

import numpy as np


def canonical_seq_for(garment, flat_pos):
    ext = flat_pos[:, :2].max(0) - flat_pos[:, :2].min(0)
    c = flat_pos[:, :2].mean(0)
    g = os.path.basename(garment)
    if g.startswith("tshirt"):
        # the neat TWO-fold: left half over right half, then hem to collar
        return [(c, np.array([-1.0, 0.0])), (c, np.array([0.0, -1.0]))]
    if g.startswith("pants"):
        return [(c, np.array([-1.0, 0.0])), (c, np.array([0.0, -1.0]))]
    if g.startswith("towel"):
        # towels get the same neat two-fold as garments (quarters)
        return [(c, np.array([-1.0, 0.0])), (c, np.array([0.0, -1.0]))]
    return [(c, np.array([1.0, 0.0]))]


def flap_corners_for(flat, pt, n2):
    d = (flat[:, :2] - pt) @ n2
    side = d > 0
    t = np.array([-n2[1], n2[0]])
    flap = np.where(side)[0]
    dd = d[flap]
    far = flap[dd > dd.max() * 0.6]
    if len(far) == 0:
        far = flap[np.argsort(dd)[-4:]]
    tt = (flat[far, :2] - pt) @ t
    g1, g2 = int(far[np.argmin(tt)]), int(far[np.argmax(tt)])
    m1 = flat[g1, :2] - 2 * ((flat[g1, :2] - pt) @ n2) * n2
    m2 = flat[g2, :2] - 2 * ((flat[g2, :2] - pt) @ n2) * n2
    q1 = int(np.argmin(((flat[:, :2] - m1) ** 2).sum(1)))
    q2 = int(np.argmin(((flat[:, :2] - m2) ** 2).sum(1)))
    return g1, g2, q1, q2


def reflect_fold(pos, pt, n2, side, gap=0.012):
    """Reflection fold that PRESERVES the flap's internal z-structure
    (critical for two-layer garments — collapsing the flap to one z-plane
    interpenetrates the layers and explodes the solver)."""
    d = (pos[:, :2] - pt) @ n2
    new = pos.copy()
    new[side, :2] -= 2 * d[side, None] * n2
    flap_z = pos[side, 2]
    base_top = pos[~side, 2].max()
    # a real fold FLIPS the flap upside down: mirror its internal z-order
    # (otherwise 2-layer seams demand impossible stretch at the crease)
    new[side, 2] = base_top + gap + (flap_z.max() - flap_z)
    return new


def morph_to(scene, cloth, target, steps=30, settle_fn=None):
    """Quasi-static teleport: interpolate particle positions to `target` over
    several sub-steps, zeroing velocities — discontinuous jumps NaN the solver
    on stitched two-layer garments."""
    import numpy as np
    start = cloth.get_particles_pos().cpu().numpy()
    if start.ndim == 3:
        start = start[0]
    for i in range(steps):
        a = (i + 1) / steps
        sm = 3 * a**2 - 2 * a**3
        cur = (1 - sm) * start + sm * target
        cloth.set_particles_pos(cur)
        cloth.set_particles_vel(np.zeros_like(cur))
        scene.step()
    cloth.set_particles_vel(np.zeros_like(target))


def morph_fold(scene, cloth, pt, n2, side, steps=50, gap=0.02):
    """Physically-plausible fold: rotate the flap about the crease line like a
    hinge (arc path — the flap stays ABOVE the base at all times), ending at
    the reflect_fold target. Zero velocities throughout (quasi-static)."""
    import numpy as np
    start = cloth.get_particles_pos().cpu().numpy()
    if start.ndim == 3:
        start = start[0]
    target = reflect_fold(start, pt, n2, side, gap=gap)
    d0 = (start[side, :2] - pt) @ n2          # crease-frame distance
    base_top = start[~side, 2].max()
    h0 = start[side, 2] - base_top            # height rel. to hinge plane
    for i in range(steps):
        a = (i + 1) / steps
        theta = np.pi * (3 * a**2 - 2 * a**3)
        cur = start.copy()
        dd = d0 * np.cos(theta) - h0 * np.sin(theta)
        hh = d0 * np.sin(theta) + h0 * np.cos(theta)
        cur[side, :2] = start[side, :2] + np.outer(dd - d0, n2)
        cur[side, 2] = base_top + np.maximum(hh, 0.004)
        if a > 0.8:                           # blend into the exact target
            b = (a - 0.8) / 0.2
            cur = (1 - b) * cur + b * target
        cloth.set_particles_pos(cur)
        cloth.set_particles_vel(np.zeros_like(cur))
        scene.step()
    cloth.set_particles_vel(np.zeros_like(target))
