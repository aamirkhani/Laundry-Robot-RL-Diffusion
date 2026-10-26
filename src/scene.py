"""Cloth-folding scene: two floor-mounted Franka Pandas + PBD towel.

build_scene() returns (scene, left_arm, right_arm, cloth, cam).
The towel spawns flat above the workspace; spawn_tangled() crumples it.
"""
import os

import numpy as np
import genesis as gs

from make_cloth_mesh import make_towel

# cotton-look material — stress-tested: 1.4e-2/0.006 survives the two-fold
# goal state plus 3 stacked scramble folds; the crisper 8e-3/0.02 explodes
# once any fold lands on top of the folded goal
BENDING = 1.4e-2
AIR = 0.006
PARTICLE_SIZE = None    # None = legacy 0.4/(n_grid-1); set ~0.0067 for 4x meshes
SUBSTEPS = None         # None = build_scene's substeps arg

TABLE_H = 0.75                  # tabletop height — everything lives on the table
TABLE_SIZE = (2.2, 1.6)         # a very large work table
GLASS_H = 0.30                  # glass enclosure walls around the tabletop rim

WORKSPACE = (0.0, 0.0)          # cloth work area center (on the tabletop)
STACK_POS = (0.0, 0.36)         # folded items get placed here
ARM_L_POS = (-0.62, 0.0, TABLE_H)   # arms mounted ON the table
ARM_R_POS = (0.62, 0.0, TABLE_H)
ARM_I_POS = (0.0, -0.58, TABLE_H)   # third arm: the ironing station
HOME_I = None                        # set below (faces +y toward the workspace)

HOME_L = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.8, 0.03, 0.03])
HOME_R = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.8, 0.03, 0.03])
HOME_I = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.8, 0.0, 0.0])


def build_scene(n_grid=31, show_viewer=False, dt=0.005, substeps=10, seed=0,
                garment=None):
    """garment: None = square towel (n_grid), or a mesh path like 'tshirt.obj'."""
    gs.init(backend=gs.gpu, seed=seed, logging_level="warning")
    psize = PARTICLE_SIZE if PARTICLE_SIZE is not None else 0.4 / (n_grid - 1)
    substeps = SUBSTEPS if SUBSTEPS is not None else substeps
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        pbd_options=gs.options.PBDOptions(particle_size=psize,
                                          lower_bound=(-0.55, -0.55, TABLE_H),
                                          upper_bound=(0.55, 0.55, TABLE_H + 1.5)),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.5, -1.5, TABLE_H + 1.1),
                                                camera_lookat=(0.0, 0.0, TABLE_H + 0.15)),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    # the big work table
    scene.add_entity(gs.morphs.Box(size=(TABLE_SIZE[0], TABLE_SIZE[1], TABLE_H),
                                   pos=(0.0, 0.0, TABLE_H / 2), fixed=True),
                     surface=gs.surfaces.Default(color=(0.45, 0.32, 0.22, 1.0)))
    # glass enclosure walls on the tabletop rim
    gx, gy = TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2
    glass = gs.surfaces.Default(color=(0.7, 0.85, 0.95, 0.25))
    for size, pos in [
        ((0.02, TABLE_SIZE[1], GLASS_H), (-gx + 0.01, 0.0, TABLE_H + GLASS_H / 2)),
        ((0.02, TABLE_SIZE[1], GLASS_H), (gx - 0.01, 0.0, TABLE_H + GLASS_H / 2)),
        ((TABLE_SIZE[0], 0.02, GLASS_H), (0.0, -gy + 0.01, TABLE_H + GLASS_H / 2)),
        ((TABLE_SIZE[0], 0.02, GLASS_H), (0.0, gy - 0.01, TABLE_H + GLASS_H / 2)),
    ]:
        scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=True,
                                       collision=False), surface=glass)

    left = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
                       pos=ARM_L_POS, euler=(0, 0, 0)),
    )
    right = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
                       pos=ARM_R_POS, euler=(0, 0, 180)),
    )

    ironer = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
                       pos=ARM_I_POS, euler=(0, 0, 90)),
    )
    # the iron: a flat plate, kinematically glued under the ironer's hand each
    # step (visual + smoothing footprint; no rigid collision)
    iron_plate = scene.add_entity(
        gs.morphs.Box(size=(0.14, 0.09, 0.02), pos=(0.0, -0.45, TABLE_H + 0.5),
                      fixed=False, collision=False),
        surface=gs.surfaces.Default(color=(0.15, 0.15, 0.18, 1.0)),
    )
    scene._ironer = ironer
    scene._iron_plate = iron_plate
    scene._scorch_pool = [
        scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.004),
                                       pos=(2.5, 2.5, 0.05 + 0.02 * i),
                                       fixed=False, collision=False),
                         surface=gs.surfaces.Default(color=(0.12, 0.08, 0.05, 1.0)))
        for i in range(8)
    ]

    if garment:
        mesh_path = garment
    else:
        mesh_path = f"towel_{n_grid}.obj"
        if not os.path.exists(mesh_path):
            make_towel(n_grid, 0.4, mesh_path)
    cloth = scene.add_entity(
        # floppy, high-friction towel: near-zero bending resistance so folds
        # persist instead of springing open; friction keeps layers from sliding
        # cotton-like: crisper bending than the "rubbery" first pass, more
        # air damping (kills jiggle), high friction
        material=gs.materials.PBD.Cloth(
            bending_compliance=BENDING,
            air_resistance=AIR,
            static_friction=0.55,
            kinetic_friction=0.45,
        ),
        morph=gs.morphs.Mesh(file=mesh_path,
                             pos=(WORKSPACE[0], WORKSPACE[1], TABLE_H + 0.35),
                             euler=(0, 0, 0)),
        surface=gs.surfaces.Default(color=(0.85, 0.35, 0.25, 1.0)),
    )

    cam = scene.add_camera(res=(1280, 720), pos=(1.5, -1.5, TABLE_H + 1.05),
                           lookat=(0.0, 0.05, TABLE_H + 0.12), fov=40, GUI=False)
    return scene, left, right, cloth, cam


def hold_home(left, right):
    left.control_dofs_position(HOME_L)
    right.control_dofs_position(HOME_R)


def settle(scene, steps=150, cam=None, record=False, every=3):
    for i in range(steps):
        scene.step()
        if record and cam is not None and i % every == 0:
            cam.render()


def spawn_tangled(scene, cloth, rng, n_folds=None, settle_steps=250, cam=None,
                  record=False, return_oracle=False):
    """Tangle the garment with random reflection folds applied ONE AT A TIME
    with damped settling between — the cotton material tolerates a single
    teleported fold + settle (proven by the collectors) but explodes if a
    whole multi-fold stack is teleported at once.
    return_oracle: also return the LAST fold's flap-corner particle indices —
    the known-correct grasps for the first un-fold (the user's hack: we made
    the tangle, so we know its inverse for free)."""
    from garment_folds_lib import reflect_fold

    def damped(steps):
        for _ in range(max(1, steps // 10)):
            settle(scene, 10, cam=cam, record=record)
            p = cloth.get_particles_pos().cpu().numpy()
            v = cloth.get_particles_vel().cpu().numpy()
            if p.ndim == 3:
                p, v = p[0], v[0]
            m = p[:, 2] < TABLE_H + 0.025
            v[m, 0] = 0.0
            v[m, 1] = 0.0
            s = np.linalg.norm(v, axis=1)
            f = s > 3.0
            if f.any():
                v[f] *= (3.0 / s[f])[:, None]
            cloth.set_particles_vel(v)

    oracle = None
    n_folds = n_folds if n_folds is not None else rng.integers(3, 5)
    for k in range(int(n_folds)):
        pos = cloth.get_particles_pos().cpu().numpy()
        if pos.ndim == 3:
            pos = pos[0]
        center = pos[:, :2].mean(axis=0)
        for _ in range(20):
            pt = center + rng.uniform(-0.06, 0.06, 2)
            ang = rng.uniform(0, np.pi)
            n2 = np.array([np.cos(ang), np.sin(ang)])
            d = (pos[:, :2] - pt) @ n2
            side = d > 0
            if side.sum() >= 30 and (~side).sum() >= 30:
                break
        else:
            continue
        if side.sum() > (~side).sum():
            side = ~side
            d = -d
        new = reflect_fold(pos, pt, n2, side)
        t2 = np.array([-n2[1], n2[0]])
        flap_idx = np.where(side)[0]
        dd = np.abs(d[flap_idx])
        far = flap_idx[dd > dd.max() * 0.6]
        if len(far) == 0:
            far = flap_idx[np.argsort(dd)[-4:]]
        tt = (new[far, :2] - pt) @ t2
        oracle = (int(far[np.argmin(tt)]), int(far[np.argmax(tt)]))
        cloth.set_particles_pos(new)
        cloth.set_particles_vel(np.zeros_like(new))
        damped(120)
    # random yaw + slight jitter (rigid rotation is isometric — safe), no drop
    pos = cloth.get_particles_pos().cpu().numpy()
    if pos.ndim == 3:
        pos = pos[0]
    theta = rng.uniform(0, 2 * np.pi)
    Rz = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    c = pos[:, :2].mean(axis=0)
    pos[:, :2] = (pos[:, :2] - c) @ Rz.T + c + rng.uniform(-0.03, 0.03, 2)
    pos[:, 2] += 0.015
    cloth.set_particles_pos(pos)
    cloth.set_particles_vel(np.zeros_like(pos))
    damped(max(80, settle_steps - 120))
    if return_oracle:
        return oracle
