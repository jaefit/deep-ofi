"""Hypothesis test: is multi-level OFI a compact sufficient statistic?

For each symbol and horizon, predict the 3-class mid-price direction from three
input sets, using the SAME classifiers (so any difference is the INPUT, not the
model):
    - ofi_best  : level-1 OFI only            (1 feature)
    - ofi_multi : all-level OFI                (L features)
    - raw       : full normalized LOB snapshot (4L features)

Three separate, pre-registered questions (split so we don't need OFI to "win"
overall to have a result):
  Q1  multi-level OFI > single-level OFI?      F1(ofi_multi) vs F1(ofi_best)
  Q2  does OFI (flow) match the raw book (state)? F1(ofi_multi) vs F1(raw)
  Q3  does OFI add ON TOP of the static book?  F1(raw_plus_ofi) vs F1(raw)

Metric: macro-F1 (3 imbalanced classes), test = last 30% in time (no shuffle).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.features import build_dataset

warnings.filterwarnings("ignore")
MET = Path(__file__).resolve().parents[1] / "results" / "metrics"
MET.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HORIZONS = [10, 20, 50, 100]
INPUTS = ["ofi_best", "ofi_multi", "raw", "raw_plus_ofi"]


def _models():
    return {
        "logistic": LogisticRegression(max_iter=500, C=1.0),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                             early_stopping=True, random_state=0),
    }


def run_one(symbol, horizon):
    ds = build_dataset(symbol, horizon=horizon)
    tr, te = ds["tr_idx"], ds["te_idx"]
    ytr, yte = ds["y"][tr], ds["y"][te]
    # trivial baseline: always predict the train-majority class
    maj = np.bincount(ytr, minlength=3).argmax()
    base_f1 = f1_score(yte, np.full_like(yte, maj), average="macro")
    rows = []
    for inp in INPUTS:
        Xtr, Xte = ds[inp][tr], ds[inp][te]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for mname, m in _models().items():
            m.fit(Xtr, ytr)
            f1 = f1_score(yte, m.predict(Xte), average="macro")
            rows.append(dict(symbol=symbol, horizon=horizon, input=inp,
                             model=mname, n_feat=Xtr.shape[1], macro_f1=float(f1)))
    return rows, float(base_f1)


def main():
    all_rows = []
    q1 = q2 = q3 = 0
    cells = 0
    for sym in SYMBOLS:
        for h in HORIZONS:
            rows, base = run_one(sym, h)
            all_rows += rows
            best = {inp: max(r["macro_f1"] for r in rows if r["input"] == inp) for inp in INPUTS}
            cells += 1
            q1 += best["ofi_multi"] >= best["ofi_best"]
            q2 += best["ofi_multi"] >= best["raw"] - 0.005
            q3 += best["raw_plus_ofi"] >= best["raw"] + 0.005
            print(f"{sym} h={h:3d} base={base:.3f} | OFI1={best['ofi_best']:.3f} "
                  f"OFImulti={best['ofi_multi']:.3f} raw={best['raw']:.3f} "
                  f"raw+OFI={best['raw_plus_ofi']:.3f}")
    print(f"\nQ1 multi>=single OFI : {q1}/{cells} cells")
    print(f"Q2 OFI>=raw (flow~state): {q2}/{cells} cells")
    print(f"Q3 raw+OFI>raw (flow adds): {q3}/{cells} cells")
    (MET / "ofi_results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"Saved -> {MET}/ofi_results.json  ({len(all_rows)} rows)")
    return all_rows


if __name__ == "__main__":
    main()
