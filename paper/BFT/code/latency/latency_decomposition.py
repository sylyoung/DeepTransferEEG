"""
latency_decomposition.py — per-sample test-time latency of T-TIME, BFT-A and
BFT-D, broken down by stage. This is the latency half of Table VIII, Section
IV-H: EA, transform generation, forward, ranking, aggregation and backprop, in
milliseconds per test sample, warmed up and with the K branches batched.

The accuracy half of Table VIII comes from BFT-classify/quantization.py.

Reproduces the exact method:
  * EEGNet F1=8,D=2,F2=16 split into block_model (conv feature) + classifier (FC), 2810 params;
  * EEGNetLossPredictor ranker 496->248->124->1 = 154,257 params (matches loss_predictor_params);
  * the exact K=12 BFT-A transform bank (identity, noise, 3x scale, 2x Hilbert freq-shift,
    5x sliding-window) and BFT-D K=10 contiguous feature-range masking.

Fixes the two measurement faults of the original quantization_utils.py:
  (1) no warm-up  -> add warm-up before timing;
  (2) the K branches are looped, and for BFT-A block_model is computed 2K times
      (ranking loop + output loop) -> compute the backbone ONCE per branch and BATCH
      the K branches into a single forward of shape (K,1,C,T).

Latency is weight-independent, so random init at the exact architecture/shape reproduces
the timing without checkpoints or data. EA and transform are CPU signal processing (as in
the paper); Forward/Ranking/Aggregation/Backprop run on the chosen compute device.
"""
import sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert
from scipy.linalg import fractional_matrix_power

# Zhou2016 / EEGNet, exactly as BFT-classify/quantization.py
CHN, TSN, SR, NCLS = 14, 1251, 250, 2
EEG_LEN = (round(TSN / SR) - 1) * SR          # 1000
K_A, DROP_SPLITS = 12, 10                     # BFT-A K=12 ; BFT-D K=10
TAU = 0.25                                    # temperature used in the original (target_output/0.25)
WARMUP, REPS = 40, 200
TTIME_BATCH = 8


# ----------------------- exact EEGNet (block + classifier) -----------------------
class EEGNet_Block(nn.Module):
    def __init__(self, chn, samples, kern, F1=8, D=2, F2=16, p=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kern // 2 - 1, kern - kern // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, kern), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (chn, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p))

    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        return x.reshape(x.size(0), -1)


class EEGNet_Classifier(nn.Module):
    def __init__(self, fdim, ncls):
        super().__init__(); self.fc = nn.Linear(fdim, ncls)

    def forward(self, x):
        return self.fc(x)


class EEGNetLossPredictor(nn.Module):
    """Exact ranker: 496 -> 248 -> 124 -> 1 (154,257 params)."""
    def __init__(self, fdim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fdim, fdim // 2), nn.ELU(),
            nn.Linear(fdim // 2, fdim // 4), nn.ELU(),
            nn.Linear(fdim // 4, 1))

    def forward(self, x):
        return self.net(x)


# ----------------------- exact BFT-A transform bank -----------------------
def _freq_shift(x, f_shift, sr, nch):           # x: (T, C)
    n = int(np.ceil(np.log2(abs(len(x)))))
    padlen = 2 ** n
    padded = np.vstack((x, np.zeros((padlen - len(x), nch))))
    h = hilbert(padded, axis=0)
    t = np.arange(padlen)
    sh = np.exp(2j * np.pi * f_shift / sr * t)
    out = np.zeros(x.shape)
    for i in range(nch):
        out[:, i] = (h[:, i] * sh)[:len(x)].real
    return out


def generate_views(x):                          # x: (1, C, TSN) -> list of 12 (C, EEG_LEN)
    xc = np.transpose(x[0], (1, 0))             # (TSN, C)
    out = [x[0, :, :EEG_LEN]]                    # identity
    std = np.std(xc)                             # noise
    out.append(np.transpose(xc + (np.random.rand(*xc.shape) - 0.5) * std / 2, (1, 0))[:, :EEG_LEN])
    for m in (0.1, -0.1, -0.2):                  # scaling x3
        out.append(np.transpose(xc * (1 - m), (1, 0))[:, :EEG_LEN])
    out.append(np.transpose(_freq_shift(xc, 0.2, SR, CHN), (1, 0))[:, :EEG_LEN])   # freq high
    out.append(np.transpose(_freq_shift(xc, -0.2, SR, CHN), (1, 0))[:, :EEG_LEN])  # freq low
    stride = int(SR * 0.2)                       # sliding window x5
    for no in (1, 2, 3, 4, 5):
        s = stride * no
        out.append(x[0, :, s:s + EEG_LEN])
    return out                                   # len 12


def ea_step(sample, R, i):
    cov = np.cov(sample)
    R = cov if i == 0 else (R * i + cov) / (i + 1)
    return fractional_matrix_power(R, -0.5).real @ sample, R


def timed(fn, sync, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ts))


def run(dev):
    is_cuda = (dev == 'cuda')
    sync = torch.cuda.synchronize if is_cuda else (lambda: None)
    kern = SR // 2
    block = EEGNet_Block(CHN, EEG_LEN, kern).to(dev).eval()
    with torch.no_grad():
        fdim = block(torch.zeros(1, 1, CHN, EEG_LEN, device=dev)).shape[1]
    clf = EEGNet_Classifier(fdim, NCLS).to(dev).eval()
    ranker = EEGNetLossPredictor(fdim).to(dev).eval()

    raw = np.random.randn(1, CHN, TSN).astype(np.float32)
    x_ea = np.random.randn(CHN, EEG_LEN).astype(np.float64)

    # ---- EA (CPU) & Transform (CPU) ----
    ea = timed(lambda: ea_step(x_ea, np.cov(x_ea), 5), lambda: None)
    tr_A = timed(lambda: generate_views(raw), lambda: None)

    # ---- BFT-A forward: ONE batched backbone pass over K branches (the fix) ----
    vs = np.stack(generate_views(raw)).astype(np.float32)          # (12, C, EEG_LEN)
    xA = torch.from_numpy(vs).unsqueeze(1).to(dev)                 # (12,1,C,EEG_LEN)
    def fwd_A():
        with torch.no_grad():
            f = block(xA)                                          # ONE call (was 2K looped)
            return clf(f)
    fA = timed(fwd_A, sync)
    with torch.no_grad():
        featA = block(xA)
    rankA = timed(lambda: ranker(featA) if True else None, sync)   # ONE ranker call over K
    def agg_A():
        with torch.no_grad():
            w = F.softmax(-ranker(featA).squeeze(), dim=0)
            p = F.softmax(clf(featA) / TAU, dim=1)
            return (p * w.unsqueeze(1)).sum(0)
    aggA = timed(agg_A, sync)
    # looped reference = original bug (2K block_model calls)
    x1 = xA[:1]
    def fwd_A_loop():
        with torch.no_grad():
            pl = []
            for j in range(K_A):
                pl.append(ranker(block(x1)))
            for k in range(K_A):
                clf(block(x1))
    fA_loop = timed(fwd_A_loop, sync)

    # ---- BFT-D: one shared backbone pass + K=10 contiguous masks, batched head ----
    xd = torch.from_numpy(raw[:, :, :EEG_LEN]).unsqueeze(1).to(dev)
    with torch.no_grad():
        D = block(xd).shape[1]
    idx = [(int(i / DROP_SPLITS * D), int((i + 1) / DROP_SPLITS * D)) for i in range(DROP_SPLITS)]
    def fwd_D():
        with torch.no_grad():
            o = block(xd)                                          # ONE backbone pass, reused
            m = o.repeat(DROP_SPLITS, 1)
            for k, (s, e) in enumerate(idx):
                m[k, s:e] = 0.0
            return clf(m)                                          # one batched head call
    fD = timed(fwd_D, sync)
    with torch.no_grad():
        o = block(xd); mD = o.repeat(DROP_SPLITS, 1)
        for k, (s, e) in enumerate(idx):
            mD[k, s:e] = 0.0
    rankD = timed(lambda: ranker(mD), sync)
    def agg_D():
        with torch.no_grad():
            w = F.softmax(-ranker(mD).squeeze(), dim=0)
            p = F.softmax(clf(mD) / TAU, dim=1)
            return (p * w.unsqueeze(1)).sum(0)
    aggD = timed(agg_D, sync)

    # ---- T-TIME: single forward + one backprop update (batch=8) ----
    net = nn.Sequential(block, clf)
    def ttf():
        with torch.no_grad():
            return clf(block(x1))
    ttime_fwd = timed(ttf, sync)
    xb = torch.randn(TTIME_BATCH, 1, CHN, EEG_LEN, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    def ttbp():
        net.train()
        out = clf(block(xb))
        sm = F.softmax(out / 2.0, dim=1)
        ent = -(sm * torch.log(sm + 1e-5)).sum(1).mean()
        msm = sm.mean(0)
        loss = ent + (msm * torch.log(msm + 1e-5)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); net.eval()
    ttime_bp = timed(ttbp, sync)

    print(f"\n===== compute device = {dev.upper()}  (EA & Transform on CPU; warmup={WARMUP}, reps={REPS}) =====")
    print(f"backbone={sum(p.numel() for p in block.parameters())+sum(p.numel() for p in clf.parameters())} "
          f"params | ranker={sum(p.numel() for p in ranker.parameters())} params (fdim={fdim})")
    hdr = f"{'method':<8}{'K':>3}{'EA':>8}{'Transf':>9}{'Forward':>9}{'Rank':>8}{'Aggreg':>8}{'Backprop':>10}{'Total':>9}"
    print(hdr)
    rows = []
    def emit(name, K, tr, fwd, rk, ag, bp):
        tot = ea + tr + fwd + rk + ag + bp
        print(f"{name:<8}{K:>3}{ea:8.2f}{tr:9.2f}{fwd:9.2f}{rk:8.2f}{ag:8.2f}{bp:10.2f}{tot:9.2f}")
        rows.append(dict(device=dev, method=name, K=K, EA=round(ea,2), Transform=round(tr,2),
                         Forward=round(fwd,2), Ranking=round(rk,2), Aggregation=round(ag,2),
                         Backprop=round(bp,2), Total=round(tot,2)))
    emit('T-TIME', 1, 0.0, ttime_fwd, 0.0, 0.0, ttime_bp)
    emit('BFT-A', K_A, tr_A, fA, rankA, aggA, 0.0)
    emit('BFT-D', DROP_SPLITS, 0.02, fD, rankD, aggD, 0.0)
    print(f"[ref] BFT-A Forward LOOPED 2K (original bug): {fA_loop:.2f} ms  vs BATCHED {fA:.2f} ms "
          f"-> {fA_loop/max(fA,1e-9):.1f}x")
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--devices', default='cpu,cuda')
    ap.add_argument('--threads', type=int, default=20)
    ap.add_argument('--csv', default='')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| threads:", torch.get_num_threads())
    all_rows = []
    for d in args.devices.split(','):
        d = d.strip()
        if d == 'cuda' and not torch.cuda.is_available():
            print("(skip cuda)"); continue
        all_rows += run(d)
    if args.csv and all_rows:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
        print("\nwrote", args.csv)
