"""Train the tangle-inversion denoiser (discrete diffusion per Khani 2026):
cross-entropy prediction of the inverse un-fold action's two grasp cells,
conditioned on the settled post-fold heightmap. The cloth state itself encodes
the noise level — no timestep conditioning (paper's key simplification).

Usage: python train_denoiser.py --data data_scramble --out denoiser.pt
"""
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HM = 16
OBS = HM * HM + 1


class Denoiser(nn.Module):
    """Small CNN over the heightmap; two grasp-cell heads (unordered pair)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head1 = nn.Conv2d(64, 1, 1)
        self.head2 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        hm = x[:, : HM * HM].reshape(-1, 1, HM, HM)
        h = self.conv(hm)
        return self.head1(h).flatten(1), self.head2(h).flatten(1)


def sym_loss(l1, l2, y1, y2):
    a = F.cross_entropy(l1, y1, reduction="none") + F.cross_entropy(l2, y2, reduction="none")
    b = F.cross_entropy(l1, y2, reduction="none") + F.cross_entropy(l2, y1, reduction="none")
    return torch.minimum(a, b).mean()


def cell_dist(pred, true):
    pi, pj = pred // HM, pred % HM
    ti, tj = true // HM, true % HM
    return ((pi - ti) ** 2 + (pj - tj) ** 2).float().sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_scramble")
    ap.add_argument("--out", type=str, default="denoiser.pt")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    Xs, Y1s, Y2s = [], [], []
    for f in sorted(glob.glob(f"{args.data}/shard_*.npz")):
        d = np.load(f)
        if d["X"].size:
            Xs.append(d["X"])
            Y1s.append(d["Y1"])
            Y2s.append(d["Y2"])
    X = np.concatenate(Xs)
    Y1 = np.concatenate(Y1s)
    Y2 = np.concatenate(Y2s)
    print(f"dataset: {X.shape[0]} pairs from {len(Xs)} shards")

    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    val = idx[: n // 10]
    trn = idx[n // 10:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Denoiser().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Y1t = torch.as_tensor(Y1, dtype=torch.long, device=dev)
    Y2t = torch.as_tensor(Y2, dtype=torch.long, device=dev)

    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(trn)
        tot = 0.0
        for k in range(0, len(perm), 256):
            b = perm[k:k + 256]
            l1, l2 = net(Xt[b])
            loss = sym_loss(l1, l2, Y1t[b], Y2t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep % 10 == 9 or ep == 0:
            net.eval()
            with torch.no_grad():
                l1, l2 = net(Xt[val])
                p1, p2 = l1.argmax(1), l2.argmax(1)
                d_a = cell_dist(p1, Y1t[val]) + cell_dist(p2, Y2t[val])
                d_b = cell_dist(p1, Y2t[val]) + cell_dist(p2, Y1t[val])
                d = torch.minimum(d_a, d_b) / 2
                cell_m = 2 * 0.35 / HM
                print(f"epoch {ep+1}: train loss {tot/len(trn):.3f} | "
                      f"val mean grasp err {float(d.mean())*cell_m*100:.1f} cm | "
                      f"within 1 cell: {float((d <= 1.0).float().mean())*100:.0f}%",
                      flush=True)

    torch.save({"net": net.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
