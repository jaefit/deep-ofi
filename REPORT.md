# Deep Order-Flow Imbalance — Results Report

**Question:** is **order-flow imbalance (OFI)** a *sufficient statistic* for the
limit order book when predicting the short-horizon mid-price direction?

**Data:** ~54,000 self-collected live L2 snapshots (Binance `depth20@100ms`,
BTCUSDT + ETHUSDT, ~27k each, ~45 min). 3-class direction label at horizons
k ∈ {10,20,50,100} snapshots (1–10 s), fixed ±0.3 bp flat band. Leak-free
temporal split (last 30% test), scaler fit on train only. Metric: macro-F1.

Three questions were fixed **before** seeing results, so the headline is a real
test, not a fishing expedition.

---

## Q1 — Does book depth help? Multi-level vs single-level OFI

Multi-level OFI ≥ single-level OFI in **5 / 8** cells — but the win is
**horizon-dependent**: at short horizons (k=10,20) the extra levels clearly help;
at the longest horizon (k=100) single-level (best-quote) OFI is as good or better
(e.g. BTC k=100: OFI-L1 0.451 vs OFI-all 0.383). Deep-book pressure carries
short-horizon signal; far-horizon moves are driven by the top of book.

## Q2 — Flow vs state: does OFI match the raw book? **No — the raw book wins.**

This is the central finding. The naive "sequence models rescue OFI" story is a
trap, and a capacity control springs it.

**Static models (logistic / MLP on a snapshot):** the **raw book wins, 0 / 8.**
The raw LOB vector contains the current **queue imbalance** — the single strongest
known short-horizon predictor — which a windowed *flow* measure does not encode.
So as a static feature, OFI is **not** a sufficient statistic. (Fig:
`results/figures/ofi_static.png`.)

| input (best static model) | BTC k=20 | BTC k=50 | ETH k=20 | ETH k=50 |
|---|---|---|---|---|
| OFI level-1 | 0.311 | 0.380 | 0.346 | 0.398 |
| OFI all levels | 0.368 | 0.373 | 0.362 | 0.400 |
| **raw LOB** | **0.404** | **0.402** | **0.432** | **0.448** |

**Sequence model (LSTM over the last 20 snapshots):** at first glance the winner
seems to **flip** — OFI seq (20-dim) > raw seq (80-dim) in 4/4 cells. But that is a
**model-capacity artifact, not a representation win**, and a control proves it. The
raw sequence has 4× the input width (80 vs 20); the small LSTM underfits it. When
we **PCA-compress the raw sequence to the same 20 dimensions** as OFI, the gap
**vanishes** (3 seeds, mean ± std):

| cell | OFI seq (20-dim) | raw seq (80-dim) | **raw seq → PCA 20-dim** |
|---|---|---|---|
| BTC k=50  | 0.399 ± 0.018 | 0.376 ± 0.015 | **0.391 ± 0.017** |
| BTC k=100 | 0.374 ± 0.008 | 0.363 ± 0.021 | **0.375 ± 0.003** |
| ETH k=50  | 0.373 ± 0.018 | 0.344 ± 0.008 | **0.363 ± 0.005** |
| ETH k=100 | 0.357 ± 0.018 | 0.326 ± 0.008 | **0.328 ± 0.016** |

At matched dimensionality OFI and raw are a **tie** (3/4 within noise; a small
residual OFI edge only at ETH k=100). So OFI is *not* a better representation — it
is roughly an information-preserving 20-dim compression of the book, no better than
PCA. **Decisive context:** the best predictors in the entire study are the
**single-snapshot raw-book models** (logistic / small MLP, 0.40–0.45), which beat
*every* LSTM (0.33–0.40) in all four overlapping cells. A model on **one** snapshot
of the book outperforms the sequence models on 20 — the raw book's instantaneous
queue-imbalance state dominates, and the deep-sequence machinery underperforms it
here. (Fig: `results/figures/ofi_sequence.png`.)

## Q3 — Is flow complementary to state? `raw + OFI` vs `raw`

For static models, adding OFI on top of the raw book helps in only **1 / 8** cells
(and marginally). At the static level, OFI's information is largely already
implied by the book state — flow is **not** complementary there.

---

## Bottom line

> **The hypothesis is refuted: OFI is not a sufficient statistic for the book, nor
> a better representation than it.** The raw order book — specifically its
> instantaneous queue-imbalance *state* — is the strongest and *simplest* predictor:
> a single-snapshot model (logistic or a small MLP — no sequence needed) beats every
> OFI model and every LSTM. OFI
> (a windowed *flow*) never wins like-for-like, adds ~nothing on top of the raw book,
> and its one apparent advantage (beating the raw *sequence* in an LSTM) disappears
> under a capacity-matched control — it was the small network underfitting the wider
> raw input, not OFI carrying more signal. OFI's real, modest merit is **economy**:
> a 20-dim flow vector roughly matches a 20-dim PCA compression of the 80-dim book.

This is a *controlled negative result*. The value isn't "ML beats the book" — it's
showing, with a pre-registered design and a capacity control, exactly **why** a
plausible "OFI is a sufficient statistic" story fails here, and distinguishing a
representation effect from a model-capacity artifact. All models do beat the
majority-class baseline (F1 ≈ 0.19–0.29 → 0.33–0.45), so the predictability is real
but small, as expected at 100 ms on liquid crypto.

## Honesty notes (consistent with the rest of this portfolio)
- **No look-ahead:** features use snapshots ≤ t; labels look at future mids; single
  temporal split; scaler fit on train only.
- **Pre-registered** Q1/Q2/Q3 and a fixed label band — no tuning until OFI "won".
- **Multi-seeded** the LSTM (the close call) and reported bands; the static models
  are deterministic.
- The collection window had a mild downward drift, so labels lean "down" at long
  horizons — this hits all inputs equally, so it doesn't bias the comparison, but
  it's why macro-F1 (not accuracy) is the metric.

## Limitations / next steps
- One ~45-min window, two crypto pairs; not a multi-day / multi-asset study.
- Snapshot-based OFI (100 ms) approximates event-based OFI; true tick data (LOBSTER)
  would sharpen it.
- Next: larger LSTM / temporal CNN (DeepLOB), event-time instead of 100 ms clock,
  and cross-venue transfer (train one symbol, test another).

## Reproduce
```bash
python -m src.collect --minutes 45 --symbols btcusdt ethusdt
python -m src.run_ofi          # static headline (Q1/Q2/Q3)
python -m src.run_ofi_lstm     # sequence footnote (multi-seed)
python -m src.plot_results     # figures
```
