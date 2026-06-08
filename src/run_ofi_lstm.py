"""Symmetric sequence-model check (footnote to the logistic/MLP headline).

A small LSTM is given, SYMMETRICALLY, either the sequence of per-step multi-level
OFI vectors or the sequence of raw LOB vectors — same architecture, same training.
This mirrors the deep-learning setup of Kolm, Turiel & Westray (2023) "Deep Order
Flow Imbalance". The architecture and input set are FIXED here in advance; we
report whatever it says (we do not tune until OFI wins).
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from src.features import build_dataset

warnings.filterwarnings("ignore")
MET = Path(__file__).resolve().parents[1] / "results" / "metrics"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HORIZONS = [50, 100]          # where there is real movement
SEQ_INPUTS = ["seq_ofi", "seq_raw"]


def _train_eval(Xtr, ytr, Xte, yte, n_feat, epochs=15, seed=0):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class Net(nn.Module):
        def __init__(self, f, h=32):
            super().__init__()
            self.lstm = nn.LSTM(f, h, batch_first=True)
            self.head = nn.Linear(h, 3)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1])

    net = Net(n_feat)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    idx = torch.randperm(len(Xtr_t))
    net.train()
    for _ in range(epochs):
        for b in range(0, len(Xtr_t), 256):
            sel = idx[b:b + 256]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[sel]), ytr_t[sel])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xte.astype(np.float32))).argmax(1).numpy()
    return float(f1_score(yte, pred, average="macro"))


SEEDS = [0, 1, 2]


def _prep(seq, tr, te, pca_dim=None):
    """Normalize a sequence (train-only); optionally PCA-compress features to
    pca_dim (fit on train timesteps) so two inputs can be compared at MATCHED
    dimensionality — controlling for model capacity, not just representation."""
    flat = seq[tr].reshape(-1, seq.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    Xtr, Xte = (seq[tr] - mu) / sd, (seq[te] - mu) / sd
    if pca_dim is not None and pca_dim < seq.shape[2]:
        from sklearn.decomposition import PCA
        p = PCA(n_components=pca_dim, random_state=0).fit(Xtr.reshape(-1, seq.shape[2]))
        Xtr = p.transform(Xtr.reshape(-1, seq.shape[2])).reshape(len(tr), seq.shape[1], pca_dim)
        Xte = p.transform(Xte.reshape(-1, seq.shape[2])).reshape(len(te), seq.shape[1], pca_dim)
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def main():
    rows = []
    for sym in SYMBOLS:
        for h in HORIZONS:
            ds = build_dataset(sym, horizon=h, sequences=True)
            tr, te = ds["tr_idx"], ds["te_idx"]
            ytr, yte = ds["y"][tr], ds["y"][te]
            L = ds["L"]
            # three inputs: OFI seq (L), raw seq (4L), and raw seq PCA-compressed to
            # L dims -> the capacity-matched control against OFI.
            specs = [("seq_ofi", "seq_ofi", None),
                     ("seq_raw", "seq_raw", None),
                     ("seq_raw_pcaL", "seq_raw", L)]
            line = f"{sym} h={h:3d} |"
            for name, key, pca in specs:
                Xtr, Xte = _prep(ds[key], tr, te, pca_dim=pca)
                f1s = np.array([_train_eval(Xtr, ytr, Xte, yte, Xtr.shape[2], seed=s)
                                for s in SEEDS])
                rows.append(dict(symbol=sym, horizon=h, input=name, model="lstm",
                                 n_feat=int(Xtr.shape[2]), macro_f1_mean=float(f1s.mean()),
                                 macro_f1_std=float(f1s.std()), seeds=f1s.tolist()))
                line += f" {name}={f1s.mean():.3f}±{f1s.std():.3f}"
            print(line, flush=True)
    (MET / "ofi_lstm_results.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved -> {MET}/ofi_lstm_results.json")
    return rows


if __name__ == "__main__":
    main()
