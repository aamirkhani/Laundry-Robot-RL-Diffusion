"""Visual fold metric: SSIM between rasterized top-down height images of the
current cloth state and the canonical folded target.

Unlike the footprint/rectangularity proxy (which scores any compact blob
well), SSIM sees the SHAPE — a true two-fold reads structurally similar to
the target; a crumpled ball does not. Rotation/translation are factored out
by aligning both images on their principal axes before comparison.

Pure numpy — rasterization from particles, windowed SSIM, no rendering.
"""
import numpy as np

RES = 64


def _principal_frame(pos):
    """Center + rotate xy so the footprint's principal axis is +x."""
    xy = pos[:, :2] - pos[:, :2].mean(0)
    cov = np.cov(xy.T)
    w, v = np.linalg.eigh(cov)
    R = v[:, ::-1].T                      # major axis first
    if np.linalg.det(R) < 0:
        R[1] *= -1
    out = xy @ R.T
    # canonical sign: heavier half of the mass on +x / +y
    if np.median(out[:, 0]) < 0:
        out[:, 0] *= -1
    if np.median(out[:, 1]) < 0:
        out[:, 1] *= -1
    return out


def rasterize(pos, table_h, half=0.30, res=RES):
    """Top-down max-height image in the principal frame, normalized."""
    xy = _principal_frame(pos)
    z = pos[:, 2] - table_h
    img = np.zeros((res, res), dtype=np.float32)
    ij = np.clip(((xy + half) / (2 * half) * res).astype(int), 0, res - 1)
    for (i, j), h in zip(ij, z):
        if h > img[i, j]:
            img[i, j] = h
    # FIXED height scale (not per-image max): a neat 4-layer fold tops out
    # near 4-6 cloth thicknesses; crumpled blobs are taller and uneven, and
    # per-image normalization was hiding exactly that difference
    return np.clip(img / 0.06, 0.0, 1.5)


def ssim(a, b, win=8, c1=0.01 ** 2, c2=0.03 ** 2):
    """Content-weighted windowed SSIM: empty-background blocks (present in
    BOTH images) would otherwise score ~1 and swamp the average — weight each
    block by how much content either image has there."""
    assert a.shape == b.shape
    n = a.shape[0] // win
    vals, wts = [], []
    for bi in range(n):
        for bj in range(n):
            pa = a[bi * win:(bi + 1) * win, bj * win:(bj + 1) * win]
            pb = b[bi * win:(bi + 1) * win, bj * win:(bj + 1) * win]
            ma, mb = pa.mean(), pb.mean()
            va, vb = pa.var(), pb.var()
            cab = ((pa - ma) * (pb - mb)).mean()
            vals.append(((2 * ma * mb + c1) * (2 * cab + c2)) /
                        ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))
            wts.append(max(ma, mb))
    vals, wts = np.array(vals), np.array(wts)
    tw = wts.sum()
    return float((vals * wts).sum() / tw) if tw > 1e-9 else 0.0


def grad_mag(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:] = img[:, 1:] - img[:, :-1]
    gy[1:, :] = img[1:, :] - img[:-1, :]
    return np.sqrt(gx ** 2 + gy ** 2)


class FoldSSIM:
    """Callable: score(pos) in [~0, 1] vs the canonical folded target.

    0.5 * height-SSIM + 0.5 * gradient-SSIM: the gradient term separates
    NEAT folds (few clean edges) from crumpled blobs (speckled roughness)
    that fool luminance-only SSIM."""

    def __init__(self, folded_goal_pos, table_h):
        self.table_h = table_h
        self.target = rasterize(folded_goal_pos, table_h)
        self.target_g = grad_mag(self.target)

    def __call__(self, pos):
        if not np.isfinite(pos).all():
            return 0.0
        img = rasterize(pos, self.table_h)
        return 0.5 * ssim(img, self.target) + \
            0.5 * ssim(grad_mag(img), self.target_g)
