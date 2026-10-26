"""Laundry scene v2: big table + glass (with a reach gap), three arms all ON
the tabletop, a bucket hanging OUTSIDE the left table rim holding crumpled
garments, fed one at a time onto the table.

build_laundry_scene(garments=[...]) returns
    (scene, left, right, ironer, cloths, cam)
cloths is a list (one entity per garment in the bucket).
"""
import numpy as np
import genesis as gs

import scene as base
from scene import TABLE_H, TABLE_SIZE, GLASS_H, HOME_L, HOME_R, HOME_I

ARM_L_POS = (-0.62, 0.0, TABLE_H)
ARM_R_POS = (0.62, 0.0, TABLE_H)
ARM_I_POS = (0.0, -0.45, TABLE_H)      # ironing arm fully ON the tabletop

# bucket: hangs off the LEFT rim (x = -1.1), aligned with the left arm so a
# pick from inside stays within Franka reach; body entirely outside the table
BUCKET_IN = 0.42                        # inner width (x and y)
BUCKET_D = 0.30                         # inner depth
BUCKET_WALL = 0.015
BUCKET_CX = -(TABLE_SIZE[0] / 2) - BUCKET_IN / 2 - BUCKET_WALL   # ~ -1.302
BUCKET_CY = 0.0
BUCKET_TOP = TABLE_H + 0.03             # rim slightly above tabletop
GLASS_GAP = 0.55                        # gap in the left glass wall (reach-through)

# folded-linen shelf: low platform at the back-right corner — the calm zone
# (bucket feeds left, ironer works front, flings churn the center)
SHELF_POS = (0.52, 0.52)
SHELF_SIZE = (0.42, 0.40)
SHELF_H = 0.02


def _add_table_veneers(scene, gx, gy):
    """COSMETIC ONLY: AI-generated albedo textures on paper-thin,
    collision=False veneer panels hugging the table's top and 4 sides.
    Nothing physical changes — the plain collision box underneath is
    untouched, so no trained model is affected. Silently skipped if the
    textures/ directory has not been generated yet."""
    import os
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "textures")
    # UV-mapped quads (Box primitives have no UVs -> textures render blank)
    faces = {
        "table_top": ("quad_top.obj", (0.0, 0.0, TABLE_H + 0.001), (0, 0, 0)),
        "side_front": ("quad_long.obj", (0.0, -gy - 0.001, TABLE_H / 2),
                       (90, 0, 0)),
        "side_back": ("quad_long.obj", (0.0, gy + 0.001, TABLE_H / 2),
                      (-90, 0, 0)),
        "side_left": ("quad_short.obj", (-gx - 0.001, 0.0, TABLE_H / 2),
                      (90, 0, 90)),
        "side_right": ("quad_short.obj", (gx + 0.001, 0.0, TABLE_H / 2),
                       (90, 0, -90)),
    }
    for name, (mesh, pos, euler) in faces.items():
        img = os.path.join(tdir, f"{name}.png")
        mp = os.path.join(tdir, mesh)
        if not (os.path.exists(img) and os.path.exists(mp)):
            continue
        scene.add_entity(
            gs.morphs.Mesh(file=mp, pos=pos, euler=euler, fixed=True,
                           collision=False),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(image_path=img)))


def build_laundry_scene(garments, dt=0.005, substeps=None, seed=0,
                        show_viewer=False):
    gs.init(backend=gs.gpu, seed=seed, logging_level="warning")
    psize = base.PARTICLE_SIZE if base.PARTICLE_SIZE is not None else 0.4 / 30
    sub = substeps if substeps is not None else (base.SUBSTEPS or 10)
    gx, gy = TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=sub),
        pbd_options=gs.options.PBDOptions(
            particle_size=psize,
            # bounds include the bucket volume (below tabletop, past the rim)
            lower_bound=(BUCKET_CX - 0.3, -0.85, BUCKET_TOP - BUCKET_D - 0.1),
            upper_bound=(0.85, 0.85, TABLE_H + 1.5)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, -1.6, TABLE_H + 1.1),
            camera_lookat=(-0.15, 0.0, TABLE_H + 0.1)),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(TABLE_SIZE[0], TABLE_SIZE[1], TABLE_H),
                                   pos=(0.0, 0.0, TABLE_H / 2), fixed=True),
                     surface=gs.surfaces.Default(color=(0.45, 0.32, 0.22, 1.0)))
    _add_table_veneers(scene, gx, gy)

    # glass enclosure; LEFT wall is split so the arm can reach the bucket
    glass = gs.surfaces.Default(color=(0.7, 0.85, 0.95, 0.25))
    seg = (TABLE_SIZE[1] - GLASS_GAP) / 2
    for size, pos in [
        ((0.02, seg, GLASS_H), (-gx + 0.01, -(GLASS_GAP + seg) / 2,
                                TABLE_H + GLASS_H / 2)),
        ((0.02, seg, GLASS_H), (-gx + 0.01, (GLASS_GAP + seg) / 2,
                                TABLE_H + GLASS_H / 2)),
        ((0.02, TABLE_SIZE[1], GLASS_H), (gx - 0.01, 0.0, TABLE_H + GLASS_H / 2)),
        ((TABLE_SIZE[0], 0.02, GLASS_H), (0.0, -gy + 0.01, TABLE_H + GLASS_H / 2)),
        ((TABLE_SIZE[0], 0.02, GLASS_H), (0.0, gy - 0.01, TABLE_H + GLASS_H / 2)),
    ]:
        scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=True,
                                       collision=False), surface=glass)

    # bucket: 4 walls + floor, rigidly "attached" (fixed) to the table rim
    bc = gs.surfaces.Default(color=(0.25, 0.28, 0.33, 1.0))
    half = BUCKET_IN / 2 + BUCKET_WALL
    wall_h = BUCKET_D + 0.03
    zc = BUCKET_TOP - wall_h / 2
    for size, pos in [
        ((BUCKET_WALL, BUCKET_IN + 2 * BUCKET_WALL, wall_h),
         (BUCKET_CX - half + BUCKET_WALL / 2, BUCKET_CY, zc)),
        ((BUCKET_WALL, BUCKET_IN + 2 * BUCKET_WALL, wall_h),
         (BUCKET_CX + half - BUCKET_WALL / 2, BUCKET_CY, zc)),
        ((BUCKET_IN + 2 * BUCKET_WALL, BUCKET_WALL, wall_h),
         (BUCKET_CX, BUCKET_CY - half + BUCKET_WALL / 2, zc)),
        ((BUCKET_IN + 2 * BUCKET_WALL, BUCKET_WALL, wall_h),
         (BUCKET_CX, BUCKET_CY + half - BUCKET_WALL / 2, zc)),
        ((BUCKET_IN + 2 * BUCKET_WALL, BUCKET_IN + 2 * BUCKET_WALL, BUCKET_WALL),
         (BUCKET_CX, BUCKET_CY, BUCKET_TOP - BUCKET_D)),
        # mounting bracket to the table rim
        ((0.10, 0.08, 0.02), (-gx - 0.05, BUCKET_CY, TABLE_H - 0.01)),
    ]:
        scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=True), surface=bc)

    # folded-linen shelf: low platform + back lip so finished stacks survive
    # nearby fling turbulence; collision=True (towels REST on it) but towels
    # only reach it after their episode scoring — no ML impact
    shelf_col = gs.surfaces.Default(color=(0.62, 0.52, 0.40, 1.0))
    scene.add_entity(gs.morphs.Box(
        size=(SHELF_SIZE[0], SHELF_SIZE[1], SHELF_H),
        pos=(SHELF_POS[0], SHELF_POS[1], TABLE_H + SHELF_H / 2), fixed=True),
        surface=shelf_col)
    scene.add_entity(gs.morphs.Box(
        size=(SHELF_SIZE[0], 0.015, 0.05),
        pos=(SHELF_POS[0], SHELF_POS[1] + SHELF_SIZE[1] / 2 - 0.008,
             TABLE_H + SHELF_H + 0.025), fixed=True), surface=shelf_col)

    left = scene.add_entity(gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda.xml", pos=ARM_L_POS, euler=(0, 0, 0)))
    right = scene.add_entity(gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda.xml", pos=ARM_R_POS, euler=(0, 0, 180)))
    ironer = scene.add_entity(gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda.xml", pos=ARM_I_POS, euler=(0, 0, 90)))
    iron_plate = scene.add_entity(
        gs.morphs.Box(size=(0.14, 0.09, 0.02), pos=(0.0, -0.30, TABLE_H + 0.5),
                      fixed=False, collision=False),
        surface=gs.surfaces.Default(color=(0.15, 0.15, 0.18, 1.0)))
    scene._ironer = ironer
    scene._iron_plate = iron_plate
    scene._scorch_pool = [
        scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.004),
                                       pos=(2.5, 2.5, 0.05 + 0.02 * i),
                                       fixed=False, collision=False),
                         surface=gs.surfaces.Default(color=(0.12, 0.08, 0.05, 1.0)))
        for i in range(8)
    ]

    mat = gs.materials.PBD.Cloth(
        bending_compliance=base.BENDING, air_resistance=base.AIR,
        static_friction=0.55, kinetic_friction=0.45)
    colors = [(0.85, 0.35, 0.25, 1.0), (0.30, 0.55, 0.85, 1.0),
              (0.40, 0.75, 0.40, 1.0)]
    cloths = []
    for i, g in enumerate(garments):
        cloths.append(scene.add_entity(
            material=mat,
            morph=gs.morphs.Mesh(file=g, pos=(0.0, 0.0, TABLE_H + 0.30 + 0.18 * i),
                                 euler=(0, 0, 0)),
            surface=gs.surfaces.Default(color=colors[i % len(colors)])))

    scene._laundry_cloths = cloths
    cam = scene.add_camera(res=(1280, 720), pos=(1.6, -1.6, TABLE_H + 1.05),
                           lookat=(-0.15, 0.0, TABLE_H + 0.1), fov=42, GUI=False)
    return scene, left, right, ironer, cloths, cam


def crumple_into_bucket(scene, cloth, rng, folds=4, settle_steps=250):
    """Crumple a garment (sequential reflect folds + noise) and drop it into
    the bucket. Returns after damped settling inside the bucket."""
    from garment_folds_lib import reflect_fold
    from metrics import particles

    def damped(steps):
        for _ in range(max(1, steps // 10)):
            base.settle(scene, 10)
            p = cloth.get_particles_pos().cpu().numpy()
            v = cloth.get_particles_vel().cpu().numpy()
            if p.ndim == 3:
                p, v = p[0], v[0]
            low = p[:, 2] < BUCKET_TOP - BUCKET_D + 0.02
            v[low, 0] = 0
            v[low, 1] = 0
            s = np.linalg.norm(v, axis=1)
            f = s > 2.5
            if f.any():
                v[f] *= (2.5 / s[f])[:, None]
            cloth.set_particles_vel(v)

    for k in range(folds):
        pos = particles(cloth)
        center = pos[:, :2].mean(0)
        pt = center + rng.uniform(-0.05, 0.05, 2)
        ang = rng.uniform(0, np.pi)
        n2 = np.array([np.cos(ang), np.sin(ang)])
        d = (pos[:, :2] - pt) @ n2
        side = d > 0
        if side.sum() > (~side).sum():
            side = ~side
        if side.sum() < 60:
            continue
        new = reflect_fold(pos, pt, n2, side)
        cloth.set_particles_pos(new)
        cloth.set_particles_vel(np.zeros_like(new))
        damped(100)
    # shrink toward center (crumple), teleport just above the current pile
    pos = particles(cloth)
    c = pos.mean(0)
    pos = c + (pos - c) * 0.7 + rng.normal(0, 0.006, pos.shape)
    c2 = pos[:, :2].mean(0)
    pile_top = BUCKET_TOP - BUCKET_D + 0.02
    for other in getattr(scene, "_laundry_cloths", []):
        if other is cloth:
            continue
        q = other.get_particles_pos().cpu().numpy()
        if q.ndim == 3:
            q = q[0]
        inb = (np.abs(q[:, 0] - BUCKET_CX) < BUCKET_IN / 2 + 0.05) &               (np.abs(q[:, 1] - BUCKET_CY) < BUCKET_IN / 2 + 0.05)
        if inb.any() and np.isfinite(q[inb, 2]).all():
            pile_top = max(pile_top, float(q[inb, 2].max()))
    pos[:, :2] += np.array([BUCKET_CX, BUCKET_CY]) - c2
    pos[:, 2] = pos[:, 2] - pos[:, 2].min() + pile_top + 0.03
    cloth.set_particles_pos(pos)
    cloth.set_particles_vel(np.zeros_like(pos))
    damped(settle_steps)
    # rescue: if it bounced out of the bucket, teleport it straight in
    q = particles(cloth)
    if not (np.isfinite(q).all()
            and abs(np.median(q[:, 0]) - BUCKET_CX) < BUCKET_IN
            and np.median(q[:, 2]) < BUCKET_TOP + 0.10):
        q = np.nan_to_num(q, nan=0.0)
        c3 = q.mean(0)
        q = c3 + (q - c3) * 0.5
        q[:, 0] += BUCKET_CX - q[:, 0].mean()
        q[:, 1] += BUCKET_CY - q[:, 1].mean()
        q[:, 2] = q[:, 2] - q[:, 2].min() + pile_top + 0.02
        cloth.set_particles_pos(q)
        cloth.set_particles_vel(np.zeros_like(q))
        damped(120)
