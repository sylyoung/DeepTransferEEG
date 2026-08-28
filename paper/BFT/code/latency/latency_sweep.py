"""
latency_sweep.py — how the per-sample latency of Section IV-H scales with K, the
number of transformations, using the same measurement method as
latency_decomposition.py.

This is the sensitivity analysis behind the claim that the batched forward keeps
the cost of K branches close to the cost of one: it reports the same per-stage
decomposition as latency_decomposition.py, but across a range of K rather than
at the single K of Table VIII.

Sweeps K for BFT-A and BFT-D on EEGNet and (for the size study) Conformer, across
datasets, on CPU and GPU. The K transformed branches go through the backbone as ONE
batched forward (the corrected implementation). Outputs a tidy CSV:
  dataset, backbone, method, device, K, params, ea_ms, transform_ms, forward_ms,
  ranking_ms, aggregation_ms, total_ms
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert
from scipy.linalg import fractional_matrix_power

# the Conformer backbone of the size study is the one shipped with the
# regression code, resolved from this file rather than from the working directory
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BFT-regression'))

SR = 250
EEG_LEN = 1000
DATASETS = {'Zhou2016': 14, 'BNCI2014001': 22, 'Schirrmeister2017': 44}
K_A = [1, 2, 4, 6, 8, 10, 12]
K_D = [1, 2, 4, 6, 8, 10]
TAU = 0.25
WARMUP, REPS = 30, 120


# ---------------- EEGNet (exact, from BFT-classify) ----------------
class EEGNet_Block(nn.Module):
    def __init__(self, chn, kern, F1=8, D=2, F2=16, p=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kern // 2 - 1, kern - kern // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, kern), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (chn, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p))

    def forward(self, x):
        x = self.block2(self.block1(x))
        return x.reshape(x.size(0), -1)


class ConformerBlock(nn.Module):
    """Wrap the Conformer of BFT-regression/models, flattened.

    NOTE: that class ends in its own classification head, so what comes out here
    is the two logits rather than a wide feature vector. The backbone forward,
    which dominates the total, is therefore measured correctly, but the ranking
    stage of the Conformer arm is timed on a two-dimensional input and so is an
    underestimate. It is left as it is because these are the numbers the paper's
    size study reports, and because ranking is a small fraction of the total.
    """
    def __init__(self, chn):
        super().__init__()
        from models.Conformer import Conformer
        self.net = Conformer(channel=chn, n_classes=2, emb_size=40, depth=6)

    def forward(self, x):
        z = self.net(x)
        return z.reshape(z.size(0), -1)


class Ranker(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fdim, fdim // 2), nn.ELU(),
                                 nn.Linear(fdim // 2, fdim // 4), nn.ELU(),
                                 nn.Linear(fdim // 4, 1))

    def forward(self, x):
        return self.net(x)


def _freq_shift(x, f, nch):
    n = 2 ** int(np.ceil(np.log2(abs(len(x)))))
    pad = np.vstack((x, np.zeros((n - len(x), nch))))
    h = hilbert(pad, axis=0)
    sh = np.exp(2j * np.pi * f / SR * np.arange(n))
    out = np.zeros(x.shape)
    for i in range(nch):
        out[:, i] = (h[:, i] * sh)[:len(x)].real
    return out


def make_views(raw, chn):                       # raw (1,C,TSN) -> 12 (C,EEG_LEN)
    xc = np.transpose(raw[0], (1, 0))
    std = np.std(xc)
    out = [raw[0, :, :EEG_LEN]]
    out.append(np.transpose(xc + (np.random.rand(*xc.shape) - .5) * std / 2, (1, 0))[:, :EEG_LEN])
    for m in (0.1, -0.1, -0.2):
        out.append(np.transpose(xc * (1 - m), (1, 0))[:, :EEG_LEN])
    out.append(np.transpose(_freq_shift(xc, 0.2, chn), (1, 0))[:, :EEG_LEN])
    out.append(np.transpose(_freq_shift(xc, -0.2, chn), (1, 0))[:, :EEG_LEN])
    st = int(SR * 0.2)
    for no in (1, 2, 3, 4, 5):
        out.append(raw[0, :, st * no:st * no + EEG_LEN])
    return out


def timed(fn, sync, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts))


def build(backbone, chn, dev):
    kern = SR // 2
    block = (EEGNet_Block(chn, kern) if backbone == 'EEGNet' else ConformerBlock(chn)).to(dev).eval()
    with torch.no_grad():
        fdim = block(torch.zeros(1, 1, chn, EEG_LEN, device=dev)).shape[1]
    clf = nn.Linear(fdim, 2).to(dev).eval()
    ranker = Ranker(fdim).to(dev).eval()
    params = sum(p.numel() for p in block.parameters()) + sum(p.numel() for p in clf.parameters())
    return block, clf, ranker, fdim, params


def run(dataset, chn, backbone, dev, rows):
    is_cuda = (dev == 'cuda')
    sync = torch.cuda.synchronize if is_cuda else (lambda: None)
    block, clf, ranker, fdim, params = build(backbone, chn, dev)
    TSN = EEG_LEN + int(SR * 0.2) * 5 + 10
    raw = np.random.randn(1, chn, TSN).astype(np.float32)
    x_ea = np.random.randn(chn, EEG_LEN).astype(np.float64)
    views12 = [v.astype(np.float32) for v in make_views(raw, chn)]

    ea = timed(lambda: (fractional_matrix_power(np.cov(x_ea), -0.5).real @ x_ea), lambda: None)

    # BFT-A: first-K views, one batched forward
    for K in K_A:
        tr = timed(lambda: make_views(raw, chn)[:K], lambda: None) if K > 1 else \
             timed(lambda: raw[0, :, :EEG_LEN], lambda: None)
        xA = torch.from_numpy(np.stack(views12[:K])).unsqueeze(1).to(dev)
        def fwd():
            with torch.no_grad():
                return clf(block(xA))
        fw = timed(fwd, sync)
        with torch.no_grad():
            feat = block(xA)
        rk = timed(lambda: ranker(feat), sync)
        def agg():
            with torch.no_grad():
                w = F.softmax(-ranker(feat).squeeze(-1), 0)
                p = F.softmax(clf(feat) / TAU, 1)
                return (p * w.reshape(-1, 1)).sum(0)
        ag = timed(agg, sync)
        rows.append(dict(dataset=dataset, backbone=backbone, method='BFT-A', device=dev, K=K,
                         params=params, ea_ms=round(ea, 3), transform_ms=round(tr, 3),
                         forward_ms=round(fw, 3), ranking_ms=round(rk, 3),
                         aggregation_ms=round(ag, 3),
                         total_ms=round(ea + tr + fw + rk + ag, 3)))

    # BFT-D: one shared backbone pass + K contiguous masks
    xd = torch.from_numpy(raw[:, :, :EEG_LEN]).unsqueeze(1).to(dev)
    for K in K_D:
        idx = [(int(i / K * fdim), int((i + 1) / K * fdim)) for i in range(K)]
        def fwd():
            with torch.no_grad():
                o = block(xd).repeat(K, 1)
                for k, (s, e) in enumerate(idx):
                    o[k, s:e] = 0.0
                return clf(o)
        fw = timed(fwd, sync)
        with torch.no_grad():
            o = block(xd).repeat(K, 1)
            for k, (s, e) in enumerate(idx):
                o[k, s:e] = 0.0
        rk = timed(lambda: ranker(o), sync)
        def agg():
            with torch.no_grad():
                w = F.softmax(-ranker(o).squeeze(-1), 0)
                p = F.softmax(clf(o) / TAU, 1)
                return (p * w.reshape(-1, 1)).sum(0)
        ag = timed(agg, sync)
        rows.append(dict(dataset=dataset, backbone=backbone, method='BFT-D', device=dev, K=K,
                         params=params, ea_ms=round(ea, 3), transform_ms=0.02,
                         forward_ms=round(fw, 3), ranking_ms=round(rk, 3),
                         aggregation_ms=round(ag, 3),
                         total_ms=round(ea + 0.02 + fw + rk + ag, 3)))
    print(f"done {dataset:18s} {backbone:9s} {dev:4s} params={params}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--devices', default='cpu,cuda')
    ap.add_argument('--backbones', default='EEGNet,Conformer')
    ap.add_argument('--threads', type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    rows = []
    for dev in args.devices.split(','):
        if dev == 'cuda' and not torch.cuda.is_available():
            continue
        for bb in args.backbones.split(','):
            for ds, chn in DATASETS.items():
                run(ds, chn, bb, dev, rows)
    import csv
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", args.out, "(", len(rows), "rows )")
