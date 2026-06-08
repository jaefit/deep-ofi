"""Figures: the static-model picture (raw > OFI) and the sequence-model flip
(OFI >= raw). Reads results/metrics/*.json written by run_ofi / run_ofi_lstm.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HORIZONS = [10, 20, 50, 100]


def _best(rows, sym, h, inp):
    vals = [r["macro_f1"] for r in rows if r["symbol"] == sym and r["horizon"] == h and r["input"] == inp]
    return max(vals) if vals else np.nan


def fig_static():
    rows = json.loads((MET / "ofi_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    inputs = [("ofi_best", "OFI level-1", "tab:green"),
              ("ofi_multi", "OFI all levels", "tab:red"),
              ("raw", "raw LOB", "tab:blue")]
    for ax, sym in zip(axes, SYMBOLS):
        for inp, lab, c in inputs:
            ys = [_best(rows, sym, h, inp) for h in HORIZONS]
            ax.plot(HORIZONS, ys, "o-", color=c, label=lab)
        base = [next((r["macro_f1"] for r in rows
                      if r["symbol"] == sym and r["horizon"] == h and r["input"] == "ofi_best"), np.nan)
                for h in HORIZONS]  # placeholder; baseline drawn separately below
        ax.set_title(f"{sym} — static models (logistic/MLP)")
        ax.set_xlabel("horizon (snapshots ahead)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (test)")
    axes[0].legend(title="input")
    fig.suptitle("Static models: the raw book (queue-imbalance STATE) beats OFI flow", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "ofi_static.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_sequence():
    rows = json.loads((MET / "ofi_lstm_results.json").read_text())
    hs = sorted({r["horizon"] for r in rows})
    labels = [f"{s}\nh={h}" for s in SYMBOLS for h in hs]

    def series(inp):
        return [next(r for r in rows if r["symbol"] == s and r["horizon"] == h and r["input"] == inp)
                for s in SYMBOLS for h in hs]

    ofi, raw, pca = series("seq_ofi"), series("seq_raw"), series("seq_raw_pcaL")
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, [r["macro_f1_mean"] for r in ofi], w, yerr=[r["macro_f1_std"] for r in ofi],
           capsize=3, color="tab:red", label="OFI seq (20-dim)")
    ax.bar(x, [r["macro_f1_mean"] for r in raw], w, yerr=[r["macro_f1_std"] for r in raw],
           capsize=3, color="tab:blue", label="raw LOB seq (80-dim)")
    ax.bar(x + w, [r["macro_f1_mean"] for r in pca], w, yerr=[r["macro_f1_std"] for r in pca],
           capsize=3, color="tab:purple", label="raw seq, PCA->20-dim (capacity-matched)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("macro-F1 (test, 3 seeds)")
    ax.set_title("LSTM: OFI beats the 80-dim raw seq, but TIES the capacity-matched 20-dim raw\n"
                 "-> the apparent gap was model capacity, not representation")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ofi_sequence.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    fig_static()
    fig_sequence()
    print(f"saved figures -> {FIG}")
