# Deep Order-Flow Imbalance — Microstructure Price Prediction

> 📊 **Results & findings:** see [REPORT.md](REPORT.md) · 📚 **Beginner-friendly walkthrough:** see [TEACHING.md](TEACHING.md)

## One-line summary
Predict the **short-horizon direction of the mid-price** from the limit order book,
and test whether **order-flow imbalance (OFI)** — a compact measure of buying vs
selling pressure — is a *sufficient statistic* for the raw book. Data is **live L2
order-book snapshots I collect myself** from Binance (BTCUSDT, ETHUSDT).

## Why Sales & Trading
Market microstructure — the dynamics of the order book — is the trading desk's
core habitat. Order-flow imbalance and price impact are the bread and butter of
execution and short-horizon alpha. This project is the most "desk-native" of the
series (after volatility forecasting and deep hedging).

## The pre-registered questions
This is a hypothesis test, split into three questions fixed BEFORE seeing results
(so we don't fish for a win):

- **Q1 — does depth help?** Multi-level OFI (all 20 levels) vs single-level (best
  quote) OFI. Expectation: more levels → more signal.
- **Q2 — flow vs state.** Does OFI (a *flow* measure: net order-book changes)
  match the raw book (a *state* snapshot, which contains queue imbalance)? May be
  NO — the raw book carries the strongest known short-horizon predictor (L1 queue
  imbalance), which a pure flow measure does not encode.
- **Q3 — is flow complementary?** Does `raw + OFI` beat `raw` alone? I.e. does
  flow add information *on top of* the static book?

## Method
- **Data:** self-collected via `src/collect.py` — Binance WebSocket
  `depth20@100ms` partial-book stream, ~45 min, ~50k snapshots across 2 symbols.
  Stored as JSONL. No vendor, no auth, reproducible by re-running the collector.
- **OFI:** Cont-Kukanov-Stoikov order-flow imbalance, computed per level from
  consecutive snapshots, then summed over a trailing window (CKS show price change
  is ~linear in windowed OFI). See `src/features.py` for the exact formula.
- **Label:** 3-class mid-price direction `k` snapshots ahead, with a fixed ±0.3 bp
  flat band (a fixed band keeps the label well-posed; a data-driven quantile band
  degenerates when the mid barely moves). Horizons k ∈ {10, 20, 50, 100} (1–10 s).
- **Models (fixed in advance):** same classifiers across all input sets so any
  difference is the *input*, not the model — logistic regression + a small MLP
  (headline), plus a small symmetric **LSTM** over sequences of OFI vs raw LOB
  (footnote, mirroring Kolm-Turiel-Westray 2023).
- **Leakage controls:** features use snapshots ≤ t; label looks at future mids;
  train/test is a single temporal cut (no shuffling); scaler fit on train only.
- **Metric:** macro-F1 (3 imbalanced classes) vs a majority-class baseline.

## References
- Cont, Kukanov, Stoikov (2014), *The Price Impact of Order Book Events*.
- Kolm, Turiel, Westray (2023), *Deep Order Flow Imbalance*, Mathematical Finance.
- Cont, Cucuringu, Zhang (2023), *Cross-Impact of Order Flow Imbalance in Equity Markets*.
- Zhang, Zohren, Roberts (2019), *DeepLOB*.

## Folder structure
```
deep-ofi/
├── src/
│   ├── collect.py        # live Binance L2 collector -> data/lob_snapshots.jsonl
│   ├── features.py       # multi-level OFI + raw LOB + leak-free labels
│   ├── run_ofi.py        # logistic/MLP comparison across input sets (headline)
│   └── run_ofi_lstm.py   # symmetric sequence LSTM (footnote)
├── data/                 # self-collected snapshots (gitignored)
├── results/{metrics}     # F1 tables (json)
└── tests/
```

## Reproduce
```bash
python -m src.collect --minutes 45 --symbols btcusdt ethusdt  # collect fresh L2
python -m src.run_ofi          # logistic/MLP headline (Q1/Q2/Q3)
python -m src.run_ofi_lstm     # LSTM footnote
```
Snapshots differ each run (live data), but the pipeline and conclusions do not.
