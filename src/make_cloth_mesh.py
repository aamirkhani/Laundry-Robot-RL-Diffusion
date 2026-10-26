"""Generate a square towel grid mesh (OBJ) at a chosen resolution."""
import numpy as np
import trimesh


def make_towel(n: int = 31, size: float = 0.4, path: str = "towel.obj"):
    """n x n vertex grid, size x size meters, in the XY plane centered at origin."""
    xs = np.linspace(-size / 2, size / 2, n)
    vs = np.array([[x, y, 0.0] for y in xs for x in xs])
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            i = r * n + c
            faces.append([i, i + 1, i + n])
            faces.append([i + 1, i + n + 1, i + n])
    mesh = trimesh.Trimesh(vertices=vs, faces=np.array(faces), process=False)
    mesh.export(path)
    return path


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    out = sys.argv[2] if len(sys.argv) > 2 else f"towel_{n}.obj"
    print(make_towel(n, 0.4, out), "vertices:", n * n)
