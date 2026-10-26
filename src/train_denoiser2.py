"""Train v2 denoisers on garment scrambles.

--goal flat   -> untangle denoiser (grasp heads used by the fling rollout)
--goal folded -> FOLD denoiser: 4 heads (grasp1, grasp2, place1, place2) —
                 reverse rollout executes fold(grasp -> place) actions,
                 driving any state toward the canonically folded garment.

Usage: python train_denoiser2.py --goal folded --out fold_denoiser.pt
"""
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HM = 16


class Denoiser4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Conv2d(64, 1, 1) for _ in range(4)])

    def forward(self, x):
        hm = x[:, : HM * HM].reshape(-1, 1, HM, HM)
        h = self.conv(hm)
        return [hd(h).flatten(1) for hd in self.heads]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_scramble2")
    ap.add_argument("--goal", type=str, default="folded", choices=["flat", "folded"])
    ap.add_argument("--garment", type=str, default="",
                    help="filter: tshirt/pants/towel; '' = all garments mixed")
    ap.add_argument("--out", type=str, default="fold_denoiser.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--torch_seed", type=int, default=0)
    args = ap.parse_args()

    pats = f"{args.data}/{args.garment or '*'}*_{args.goal}_*.npz"
    Xs, G1, G2, P1, P2 = [], [], [], [], []
    for f in sorted(glob.glob(pats)):
        d = np.load(f)
        if d["X"].size:
            Xs.append(d["X"]); G1.append(d["G1"]); G2.append(d["G2"])
            P1.append(d["P1"]); P2.append(d["P2"])
    X = np.concatenate(Xs)
    Y = [np.concatenate(a) for a in (G1, G2, P1, P2)]
    # drop solver-explosion samples (heightmap cell above any plausible stack)
    keep = X[:, :HM * HM].max(1) <= 0.30 / 0.08
    X = X[keep]
    Y = [y[keep] for y in Y]
    print(f"goal={args.goal} garment={args.garment or 'ALL'}: {len(X)} pairs "
          f"from {len(Xs)} shards ({int((~keep).sum())} exploded dropped)")

    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    val, trn = idx[: n // 10], idx[n // 10:]

    dev = "cuda"
    torch.manual_seed(args.torch_seed)
    net = Denoiser4().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Yt = [torch.as_tensor(y, dtype=torch.long, device=dev) for y in Y]

    # --- refined loss: Gaussian soft targets over cells (neighbouring cells
    # are partially correct), late-stage weighting (samples near the goal get
    # extra weight -> crisp final actions), unordered-pair symmetry ---
    ii, jj = torch.meshgrid(torch.arange(HM), torch.arange(HM), indexing="ij")
    grid = torch.stack([ii.flatten(), jj.flatten()], 1).float().to(dev)

    def soft_target(cells, sigma=0.8):
        ci = (cells // HM).float().unsqueeze(1)
        cj = (cells % HM).float().unsqueeze(1)
        d2 = (grid[:, 0] - ci) ** 2 + (grid[:, 1] - cj) ** 2
        t = torch.exp(-d2 / (2 * sigma ** 2))
        return t / t.sum(1, keepdim=True)

    def soft_ce(logits, cells):
        return -(soft_target(cells) * F.log_softmax(logits, 1)).sum(1)

    def batch_loss(b):
        logits = net(Xt[b])
        w = 1.0 + 1.5 * Xt[b][:, -1]          # coverage-weighted (near-goal ↑)
        a = sum(soft_ce(logits[i], Yt[i][b]) for i in range(4))
        swap = [1, 0, 3, 2]
        bsw = sum(soft_ce(logits[i], Yt[swap[i]][b]) for i in range(4))
        return (w * torch.minimum(a, bsw)).mean()

    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(trn)
        tot = 0.0
        for k in range(0, len(perm), 256):
            b = perm[k:k + 256]
            loss = batch_loss(b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 20 == 19 or ep == 0:
            net.eval()
            with torch.no_grad():
                logits = net(Xt[val])
                accs = []
                for i in range(4):
                    pred = logits[i].argmax(1)
                    pi, pj = pred // HM, pred % HM
                    ti, tj = Yt[i][val] // HM, Yt[i][val] % HM
                    d = (((pi - ti) ** 2 + (pj - tj) ** 2).float().sqrt())
                    accs.append(float((d <= 1.5).float().mean()))
            print(f"epoch {ep+1}: loss {tot/len(perm):.3f} | "
                  f"heads within1.5cells g1 {accs[0]:.2f} g2 {accs[1]:.2f} "
                  f"p1 {accs[2]:.2f} p2 {accs[3]:.2f}", flush=True)

    torch.save({"net": net.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
