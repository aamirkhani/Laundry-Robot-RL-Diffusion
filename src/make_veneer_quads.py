"""UV-mapped quad meshes for the table texture veneers (Box primitives have
no UVs, so diffuse textures render blank on them)."""
import numpy as np
import trimesh


def quad(w, h, path):
    v = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0],
                  [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    trimesh.Trimesh(vertices=v, faces=f, process=False,
                    visual=trimesh.visual.TextureVisuals(uv=uv)).export(path)


if __name__ == "__main__":
    quad(2.2, 1.6, "textures/quad_top.obj")
    quad(2.2, 0.75, "textures/quad_long.obj")
    quad(1.6, 0.75, "textures/quad_short.obj")
    print("veneer quads written")
