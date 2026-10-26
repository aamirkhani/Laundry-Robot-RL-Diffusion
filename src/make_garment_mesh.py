"""Procedural flat garment meshes (t-shirt, pants) as mask-based triangulated
surfaces — every filled 2D grid cell becomes two triangles with shared
vertices, so the PBD sampler sees one connected cloth sheet.
"""
import numpy as np
import trimesh

CELL = 0.02


def mesh_from_mask(mask, cell=CELL, path="garment.obj"):
    """mask: 2D bool array (rows = y, cols = x). Origin centered."""
    h, w = mask.shape
    # vertex grid (only vertices adjacent to a filled cell are kept)
    vid = -np.ones((h + 1, w + 1), dtype=int)
    verts = []
    faces = []

    def vertex(r, c):
        if vid[r, c] < 0:
            vid[r, c] = len(verts)
            verts.append([(c - w / 2) * cell, (r - h / 2) * cell, 0.0])
        return vid[r, c]

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            v00 = vertex(r, c)
            v01 = vertex(r, c + 1)
            v10 = vertex(r + 1, c)
            v11 = vertex(r + 1, c + 1)
            faces.append([v00, v01, v10])
            faces.append([v01, v11, v10])
    m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=False)
    m.export(path)
    return path, len(verts)


def tshirt_mask(cell=CELL):
    """T-shirt ~0.44 wide (with sleeves) x 0.52 long, body 0.32 wide."""
    w, h = int(0.44 / cell), int(0.52 / cell)
    m = np.zeros((h, w), dtype=bool)
    bw = int(0.32 / cell)              # body width
    x0 = (w - bw) // 2
    m[: int(0.44 / cell), x0:x0 + bw] = True           # body (hem at row 0)
    sl_h0 = int(0.30 / cell)                            # sleeves near shoulders
    sl_h1 = int(0.44 / cell)
    m[sl_h0:sl_h1, :x0] = True                          # left sleeve
    m[sl_h0:sl_h1, x0 + bw:] = True                     # right sleeve
    # neck notch
    nk = int(0.06 / cell)
    cx = w // 2
    m[h - int(0.08 / cell):, cx - nk // 2: cx + nk // 2] = False
    m[int(0.44 / cell):h - int(0.08 / cell), x0:x0 + bw] = True   # shoulders
    return m


def pants_mask(cell=CELL):
    """Pants ~0.30 wide x 0.56 long; two legs below the waist block."""
    w, h = int(0.30 / cell), int(0.56 / cell)
    m = np.zeros((h, w), dtype=bool)
    waist_h = int(0.18 / cell)
    m[h - waist_h:, :] = True                           # waist/hip block (top)
    leg_w = int(0.13 / cell)
    m[: h - waist_h, :leg_w] = True                     # left leg
    m[: h - waist_h, w - leg_w:] = True                 # right leg
    return m


if __name__ == "__main__":
    p, n = mesh_from_mask(tshirt_mask(), path="tshirt.obj")
    print(p, n, "verts")
    p, n = mesh_from_mask(pants_mask(), path="pants.obj")
    print(p, n, "verts")


def mesh_from_mask_2layer(mask, cell=CELL, thickness=0.010, path="garment2.obj",
                          open_bottom=True):
    """TWO-layer garment: front + back panels stitched along the boundary
    seams; the bottom (hem) stays open so the layers read as a real shirt."""
    h, w = mask.shape
    verts = []
    vid = {}

    def vertex(r, c, layer):
        key = (r, c, layer)
        if key not in vid:
            vid[key] = len(verts)
            verts.append([(c - w / 2) * cell, (r - h / 2) * cell,
                          layer * thickness])
        return vid[key]

    faces = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for layer in (0, 1):
                v00 = vertex(r, c, layer)
                v01 = vertex(r, c + 1, layer)
                v10 = vertex(r + 1, c, layer)
                v11 = vertex(r + 1, c + 1, layer)
                if layer == 0:
                    faces.append([v00, v01, v10])
                    faces.append([v01, v11, v10])
                else:
                    faces.append([v00, v10, v01])
                    faces.append([v01, v10, v11])
    # seam walls along boundary edges (skip the hem = bottom-most edges)
    hem_rows = 2
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            edges = []
            if r == 0 or not mask[r - 1, c]:
                if not (open_bottom and r < hem_rows):
                    edges.append(((r, c), (r, c + 1)))
            if r == h - 1 or not mask[r + 1, c]:
                edges.append(((r + 1, c), (r + 1, c + 1)))
            if c == 0 or not mask[r, c - 1]:
                edges.append(((r, c), (r + 1, c)))
            if c == w - 1 or not mask[r, c + 1]:
                edges.append(((r, c + 1), (r + 1, c + 1)))
            for (ra, ca), (rb, cb) in edges:
                a0 = vertex(ra, ca, 0)
                b0 = vertex(rb, cb, 0)
                a1 = vertex(ra, ca, 1)
                b1 = vertex(rb, cb, 1)
                faces.append([a0, b0, a1])
                faces.append([b0, b1, a1])
    m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces),
                        process=False)
    m.export(path)
    return path, len(verts)
