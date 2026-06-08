"""Order-flow-imbalance features + leak-free direction labels from L2 snapshots.

Multi-level OFI follows Cont, Kukanov & Stoikov (2014) "The Price Impact of Order
Book Events", extended per-level (Xu/Cont multi-level OFI). For each book level m,
between consecutive snapshots n-1 -> n:

    bid order flow  OF^b = { +q^b_n           if P^b_n > P^b_{n-1}    (bid improved)
                             q^b_n - q^b_{n-1} if P^b_n = P^b_{n-1}
                             -q^b_{n-1}        if P^b_n < P^b_{n-1} }
    ask order flow  OF^a = { -q^a_{n-1}        if P^a_n > P^a_{n-1}
                             q^a_n - q^a_{n-1} if P^a_n = P^a_{n-1}
                             +q^a_n            if P^a_n < P^a_{n-1}    (ask improved) }
    OFI_m,n = OF^b - OF^a

Intuition: OFI > 0 = net buying pressure. CKS show price change over an interval is
approximately linear in the OFI summed over that interval — that is exactly the
predictor we build (windowed OFI sum).

Leakage guard: features at time t use snapshots <= t only; the label looks at
FUTURE mids (t+1..t+k). Train/test is a single temporal cut, scaler fit on train.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def load_snapshots(symbol: str, path: Path | None = None):
    """Return time-ordered arrays for one symbol: t, bidP/bidQ/askP/askQ (N, L), mid."""
    path = path or DATA / "lob_snapshots.jsonl"
    ts, bP, bQ, aP, aQ = [], [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["s"] != symbol:
                continue
            b = np.asarray(r["b"], dtype=float)  # (L,2) price,qty  (bids desc)
            a = np.asarray(r["a"], dtype=float)  # (L,2)            (asks asc)
            ts.append(r["t"]); bP.append(b[:, 0]); bQ.append(b[:, 1])
            aP.append(a[:, 0]); aQ.append(a[:, 1])
    t = np.asarray(ts)
    bP, bQ, aP, aQ = map(lambda x: np.asarray(x), (bP, bQ, aP, aQ))
    mid = (bP[:, 0] + aP[:, 0]) / 2.0
    return dict(t=t, bP=bP, bQ=bQ, aP=aP, aQ=aQ, mid=mid)


def multi_level_ofi(d) -> np.ndarray:
    """Per-level OFI, shape (N-1, L). Row n aligns to snapshot n (uses n-1->n)."""
    bP, bQ, aP, aQ = d["bP"], d["bQ"], d["aP"], d["aQ"]
    bP0, bP1 = bP[:-1], bP[1:]
    bQ0, bQ1 = bQ[:-1], bQ[1:]
    aP0, aP1 = aP[:-1], aP[1:]
    aQ0, aQ1 = aQ[:-1], aQ[1:]
    of_b = np.where(bP1 > bP0, bQ1, np.where(bP1 == bP0, bQ1 - bQ0, -bQ0))
    of_a = np.where(aP1 > aP0, -aQ0, np.where(aP1 == aP0, aQ1 - aQ0, aQ1))
    return of_b - of_a  # (N-1, L)


def _window_sum(ofi: np.ndarray, W: int) -> np.ndarray:
    """Trailing sum of OFI over the last W steps (causal). Returns (N-1, L)."""
    csum = np.cumsum(ofi, axis=0)
    out = csum.copy()
    out[W:] = csum[W:] - csum[:-W]
    return out


def raw_lob_features(d) -> np.ndarray:
    """Standard normalized LOB vector per snapshot: for each level
    (ask_price-mid, ask_qty, mid-bid_price, bid_qty). Shape (N, 4L)."""
    mid = d["mid"][:, None]
    ask_dp = d["aP"] - mid
    bid_dp = mid - d["bP"]
    feats = np.concatenate([ask_dp, d["aQ"], bid_dp, d["bQ"]], axis=1)
    return feats


def build_dataset(symbol: str, horizon: int = 20, window: int = 20,
                  alpha_bps: float = 0.3, train_frac: float = 0.7, path=None,
                  sequences: bool = False):
    """Build leak-free features + 3-class direction label for one symbol.

    label = sign of (mean future mid over t+1..t+k  -  mid_t) / mid_t, with a FIXED
    +/- alpha_bps basis-point flat band (a fixed band keeps the label well-posed;
    a quantile band degenerates when the mid barely moves, e.g. BTC at short
    horizons). Class balance is reported, not forced.

    Inputs returned (all causal, snapshot <= t):
      ofi_best (1), ofi_multi (L), raw (4L), raw_plus_ofi (4L+L).
    If sequences=True, also seq_ofi (M,W,L) and seq_raw (M,W,4L) for the LSTM.
    """
    d = load_snapshots(symbol, path)
    ofi = multi_level_ofi(d)                  # (N-1, L)
    raw = raw_lob_features(d)[1:]             # align to ofi rows (snapshot n)
    mid = d["mid"][1:]                        # align
    N = len(mid)
    L = ofi.shape[1]
    ofi_win = _window_sum(ofi, window)        # (N, L) trailing OFI sum

    # forward label: mean of next `horizon` mids vs current
    fut = np.full(N, np.nan)
    csum = np.cumsum(mid)
    for t in range(N - horizon):
        fut[t] = (csum[t + horizon] - csum[t]) / horizon  # mean(mid[t+1..t+h])
    pct = (fut - mid) / mid

    valid = np.arange(window, N - horizon)    # need W past + h future
    split = int(len(valid) * train_frac)
    tr_idx, te_idx = valid[:split], valid[split:]

    alpha = alpha_bps * 1e-4                   # fixed basis-point band
    y = np.where(pct > alpha, 2, np.where(pct < -alpha, 0, 1))   # 0 down,1 flat,2 up

    out = dict(
        ofi_multi=ofi_win, ofi_best=ofi_win[:, :1], raw=raw,
        raw_plus_ofi=np.concatenate([raw, ofi_win], axis=1),
        y=y, pct=pct, tr_idx=tr_idx, te_idx=te_idx, alpha=alpha, L=L, mid=mid,
        symbol=symbol, horizon=horizon, window=window,
    )
    if sequences:
        # per-step (not summed) OFI and raw, as length-W causal windows ending at t
        seq_ofi = np.stack([ofi[t - window:t] for t in range(N)], axis=0) \
            if False else _windows(ofi, window, N)
        seq_raw = _windows(raw, window, N)
        out["seq_ofi"], out["seq_raw"] = seq_ofi, seq_raw
    return out


def _windows(x: np.ndarray, W: int, N: int) -> np.ndarray:
    """Causal length-W windows ending at each t: returns (N, W, F). Rows t<W are
    zero-padded at the front (they are excluded from valid indices anyway)."""
    F = x.shape[1]
    out = np.zeros((N, W, F), dtype=np.float32)
    for t in range(W, N):
        out[t] = x[t - W:t]
    return out


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    ds = build_dataset(sym)
    yt = ds["y"][ds["tr_idx"]]
    print(f"{sym}: N_valid={len(ds['tr_idx'])+len(ds['te_idx'])} L={ds['L']} "
          f"alpha={ds['alpha']:.2e}")
    print(f"  train class balance (down/flat/up): "
          f"{np.bincount(yt, minlength=3) / len(yt)}")
