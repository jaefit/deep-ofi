"""Collect live Binance L2 order-book snapshots (top-20) to JSONL.

Combined partial-book-depth stream at 100ms for several symbols. Each line:
    {"t": <local recv epoch s>, "s": "BTCUSDT", "b": [[price,qty]*20], "a": [...]}

Robust to drops: reconnects on any error until the duration elapses. This is our
own self-collected market data — no vendor, no auth, fully reproducible by anyone
who reruns it (the exact snapshots differ, but the pipeline does not).

Usage: python -m src.collect --minutes 45 --symbols btcusdt ethusdt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)


async def run(symbols, minutes, out_path, stat_path):
    streams = "/".join(f"{s}@depth20@100ms" for s in symbols)
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    deadline = time.time() + minutes * 60
    n = 0
    with open(out_path, "w") as f:
        while time.time() < deadline:
            try:
                async with websockets.connect(url, ping_interval=15, max_queue=None) as ws:
                    while time.time() < deadline:
                        raw = await asyncio.wait_for(ws.recv(), timeout=20)
                        msg = json.loads(raw)
                        data = msg.get("data", {})
                        sym = msg.get("stream", "").split("@")[0].upper()
                        b, a = data.get("bids"), data.get("asks")
                        if not b or not a:
                            continue
                        f.write(json.dumps({"t": time.time(), "s": sym, "b": b, "a": a}) + "\n")
                        n += 1
                        if n % 500 == 0:
                            f.flush()
                            stat_path.write_text(json.dumps(
                                {"n": n, "elapsed_s": round(time.time() - (deadline - minutes * 60), 1),
                                 "remaining_s": round(deadline - time.time(), 1)}))
            except Exception as exc:  # noqa: BLE001 — reconnect on any drop
                print(f"  reconnect after: {repr(exc)[:80]}", flush=True)
                await asyncio.sleep(2)
        f.flush()
    stat_path.write_text(json.dumps({"n": n, "done": True}))
    print(f"collected {n} snapshots -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--symbols", nargs="*", default=["btcusdt", "ethusdt"])
    a = ap.parse_args()
    out = DATA / "lob_snapshots.jsonl"
    stat = DATA / "collect_status.json"
    print(f"collecting {a.symbols} for {a.minutes} min -> {out}", flush=True)
    asyncio.run(run(a.symbols, a.minutes, out, stat))


if __name__ == "__main__":
    main()
