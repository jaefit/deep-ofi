# Order-Flow Imbalance, Explained Simply

*For a 2nd-year undergrad. Easy English. No finance background needed.*

## The setup in plain words

An **order book** is the live list of buy and sell offers for an asset: at each
price, how many shares/coins people want to buy (bids) or sell (asks). The
**mid-price** is the middle of the best buy and best sell. It wiggles constantly.

**Order-flow imbalance (OFI)** is a number that summarizes *pressure*: if buyers
are piling in and sellers pulling back, OFI is positive (price likely to tick up);
the reverse makes it negative. It is computed from how the book *changes* between
two snapshots — it is a measure of **flow** (change over time).

The raw book itself is a **state** (a photo at one instant). One especially
powerful piece of that state is **queue imbalance**: are there way more shares
waiting to buy than to sell right now?

## The question we tested

> Is OFI (flow) a **sufficient statistic** — i.e. does it capture everything the
> full order book tells you about where the price goes next?

If yes, you could throw away the big book and keep one tidy number per level. That
would be elegant and practical.

## How we did it

- **Data:** I collected my own — ~54,000 live order-book snapshots from Binance
  (Bitcoin and Ethereum), 10 snapshots/second for ~45 minutes. No vendor, no
  account. (`src/collect.py`)
- **Target:** will the mid-price be **up, down, or flat** a few seconds from now?
- **The honest rules** (same as the other projects):
  - *No peeking:* features only use the past; we test on a later time slice.
  - *Pre-register:* we wrote down the exact questions before looking at results.
  - *Same model for each input:* so any difference is the *input*, not the model.
- We compared three inputs: best-level OFI, all-level OFI, and the raw book — with
  simple models (logistic regression, small neural net) and a small LSTM.

## What we found

1. **More book levels help — but only at very short horizons.** Deep-book pressure
   matters for the next second; it fades for longer horizons.
2. **OFI is NOT a sufficient statistic. The raw book wins.** In fact the single
   best predictor was one of the *simplest* ones: a plain model on a single
   snapshot of the raw book. Its queue-imbalance state is just very informative.
3. **A trap we avoided.** A small LSTM made OFI *look* better than the raw book —
   but only because the raw input was 4× bigger and the little network couldn't
   digest it. When we shrank the raw book to the same size (with PCA), the
   advantage **disappeared**. So OFI wasn't a better representation; the network
   was just underfitting the bigger input. We caught this with a **control
   experiment**, not by eyeballing.

## The lesson

> A clean *negative* result, properly controlled, is real science.

We did **not** find a magic feature. We found that the obvious-looking "OFI wins"
story falls apart under a fair test, and we showed **exactly why** (it was model
capacity, not information). Knowing the difference between "my method is better"
and "my model was just smaller and fit better" is the whole game.

## Mini-glossary
- **Order book / bid / ask:** live buy and sell offers.
- **Mid-price:** middle of best bid and best ask.
- **OFI (order-flow imbalance):** a number for net buying vs selling *pressure*, from how the book changes.
- **Queue imbalance:** how lopsided the front of the book is right now (state).
- **Macro-F1:** a fair accuracy score when the three classes are unbalanced.
- **PCA:** a way to shrink many numbers into a few without losing much — used here as a fair-size control.
- **Capacity artifact:** when a result is about the *model's size*, not the *data's information*.
