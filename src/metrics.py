"""Cloth-state metrics — order-free (PBD resamples particles, so no grid
correspondence). Based on footprint occupancy and PCA shape statistics."""
import numpy as np

TOWEL_SIZE = 0.4
FULL_AREA = TOWEL_SIZE * TOWEL_SIZE
CELL = 0.02


def particles(cloth):
    p = cloth.get_particles_pos().cpu().numpy()
    return p[0] if p.ndim == 3 else p


def footprint_cells(pos, cell=CELL):
    return set(map(tuple, np.floor(pos[:, :2] / cell).astype(int)))


def footprint_area(pos, cell=CELL):
    return len(footprint_cells(pos, cell)) * cell * cell


def coverage(pos):
    """Fraction of full flat area covered (1.0 = flat, <0.5 = crumpled)."""
    return footprint_area(pos) / FULL_AREA


def pca_axes(pos):
    xy = pos[:, :2]
    c = xy.mean(0)
    u, s, vt = np.linalg.svd(xy - c, full_matrices=False)
    return c, vt[0], vt[1]


def find_corners(pos):
    """4 corner points of a (roughly flat) towel via PCA extremes.
    Returns array (4,2): [++, +-, -+, --] along principal axes."""
    c, e1, e2 = pca_axes(pos)
    xy = pos[:, :2] - c
    a, b = xy @ e1, xy @ e2
    out = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            out.append(pos[np.argmax(s1 * a + s2 * b), :2])
    return np.array(out)  # [+e1+e2, +e1-e2, -e1+e2, -e1-e2]


def rectangularity(pos):
    """Footprint area / PCA bounding-box area (1.0 = perfect rectangle)."""
    c, e1, e2 = pca_axes(pos)
    xy = pos[:, :2] - c
    a, b = xy @ e1, xy @ e2
    box = (a.max() - a.min()) * (b.max() - b.min())
    return footprint_area(pos) / max(box, 1e-6)


def fold_quality(pos, n_grid=None):
    """Half-fold quality in [0,1]: footprint should be ~half the flat area,
    shape should stay rectangular, and the cloth should be 2 layers thick.
    Composite of area-ratio, rectangularity, and height terms."""
    ratio = footprint_area(pos) / FULL_AREA
    area_term = np.clip(1.0 - abs(ratio - 0.5) / 0.5, 0, 1)
    rect_term = np.clip(rectangularity(pos), 0, 1)
    h = pos[:, 2].max()
    height_term = np.clip(1.0 - abs(h - 0.02) / 0.10, 0, 1)
    return float(0.5 * area_term + 0.3 * rect_term + 0.2 * height_term)


def on_stack(pos, stack_xy, radius=0.22):
    center = pos[:, :2].mean(0)
    return bool(np.linalg.norm(center - np.asarray(stack_xy)) < radius)


def report(cloth, n_grid, stack_xy):
    p = particles(cloth)
    return {
        "coverage": round(coverage(p), 3),
        "fold_quality": round(fold_quality(p), 3),
        "rectangularity": round(rectangularity(p), 3),
        "on_stack": on_stack(p, stack_xy),
        "height": round(float(p[:, 2].max()) - 0.75, 3),  # above tabletop
    }
