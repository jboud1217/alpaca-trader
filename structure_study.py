"""Structure study: is there a put-spread SHAPE that earns a premium net of cost?

Every earlier sweep in this repo moved exit rules or one structural knob at a
time, measured raw P&L, and found nothing distinguishable from fair value. Two
things were missing, and this study exists to supply them.

1. DIRECTION WAS NEVER TAKEN OUT. A short put spread is long the underlying.
   Over a window where SPY rose, every short put "works", and higher delta works
   better, purely as beta -- the 2024 vendor check showed exactly that shape.
   An edge has to survive removing the directional exposure, so every trade is
   also scored DELTA-HEDGED: P&L minus entry position delta x the underlying's
   move to the exit. What is left is the premium (and gamma), not the market.

2. THE STRUCTURE ITSELF WAS NEVER SWEPT. The literature says the long wing of a
   put spread is the most overpriced leg in percentage terms (Bondarenko 2014;
   Constantinides, Jackwerth & Savov 2013), and that near-the-money short
   options are the better sale per unit of risk (Israelov & Tummala 2017). Both
   predict that a WIDER spread with a NEARER short strike keeps more of the
   premium than the deployed 0.20-delta / 3-wide shape. Width here is a
   fraction of spot, not dollars, so the three underlyings are comparable.

PRE-REGISTERED -- fixed before the first run, not after:

    grid      short delta {0.20, 0.30, 0.40, 0.50}
              wing width  {0.5%, 1%, 2%, 4%} of spot
              target DTE  {7, 14, 30}
              48 cells x SPY, QQQ, IWM
    exits     hold to expiry (no target, no stop) -- the live rule since
              2026-10-23, and the one that removes exit friction
    entries   one per day per symbol, overlapping (a daily ladder)
    window    2025-02-10 .. 2026-08-07, which contains the Feb-Apr 2025
              drawdown; a window without a sell-off cannot test put selling
    fills     measured per-symbol spread surface, slippage 0.5 (primary)
              and 1.0 (robustness)
    metric    delta-hedged P&L per dollar of max loss, pooled across symbols
    error     block bootstrap by calendar month, 2000 resamples -- daily
              ladders overlap and the three symbols co-move, so trades are not
              independent and a naive t would be several times too large

A cell QUALIFIES only if ALL of these hold:
    a. pooled hedged return-on-risk t >= 3.0  (48 cells: ~Bonferroni, one-sided)
    b. hedged return-on-risk > 0 on every symbol separately
    c. > 0 in both halves of the window
    d. still > 0 at slippage 1.0
Anything less is reported, and is not an edge.

RESULT (first run, 2026-10-23): 0 of 48 cells qualify. Highest pooled hedged
t is +1.48 (0.20 delta, 0.5% wide, 14 DTE) against ~2.8 expected from the max
of 48 noise draws. Raw return-on-risk is positive in 44 of 48 cells and hedged
in 14: most of what a short put spread "earned" here was the market rising.
    * The deployed shape (0.20 / ~1% / 6-7 DTE) hedges to -0.20%: fair value,
      matching the live hold-to-expiry baseline of -$1.30/trade.
    * Nearer the money is WORSE, not better: every 0.40 and 0.50 delta cell
      is <= 0 hedged at 7 and 30 DTE, several at t < -2. The Israelov-Tummala
      prediction does not hold on ETF options over this window.
    * Wider wings help at fixed delta (0.5% wide is the worst width in 11 of
      12 delta x DTE rows), as the overpriced-wing literature predicts.
    * 14 DTE at 0.20-0.30 delta with 1-4% wings is the one consistent pocket:
      hedged +1.2% .. +3.2%, positive on all three symbols in 6 of those 8
      cells, positive at slippage 1.0. Not significant (t 0.4-1.5), mostly
      second-half, and the cells share trades -- a hypothesis for a forward
      test, not a finding.
Caveat: the hedge is STATIC (entry delta held to exit), so near-the-money
cells carry gamma P&L a daily-rebalanced hedge would remove.

Run:  python structure_study.py            (needs APCA_API_KEY_ID / SECRET)
      python structure_study.py --quick    (one symbol, one cell; timing check)

Not financial advice.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import date

from engine import Backtest, BacktestConfig, FillModel
from spread import SpreadModel
from strategies import PutCreditSpread

START, END = date(2025, 2, 10), date(2026, 8, 7)
SYMBOLS = ("SPY", "QQQ", "IWM")
DELTAS = (0.20, 0.30, 0.40, 0.50)
WIDTHS = (0.005, 0.01, 0.02, 0.04)
DTES = (7, 14, 30)
T_QUALIFY = 3.0
BOOT = 2000


class PctWidthPutSpread(PutCreditSpread):
    """PutCreditSpread with the wing set as a fraction of spot at entry, and the
    long leg's delta recorded so the position delta can be hedged out."""

    name = "put_spread_pct"

    def __init__(self, width_pct: float, **kw):
        super().__init__(wing_width=1.0, **kw)
        self.width_pct = width_pct

    def propose_entry(self, chain, open_count):
        self.wing_width = max(1.0, round(self.width_pct * chain.spot))
        pos = super().propose_entry(chain, open_count)
        if pos is None:
            return None
        lo = min(pos.legs, key=lambda l: l.strike)
        q = chain.find("put", lo.strike, lo.expiry)
        if q is None:
            return None
        pos.meta["long_delta"] = q.delta
        return pos


def _trades(src, symbol, delta, width, dte, slip):
    strat = PctWidthPutSpread(width, short_delta=delta, target_dte=dte,
                              profit_take=None, stop_mult=None, min_dte=0)
    cfg = BacktestConfig(underlying=symbol, start=START, end=END,
                         starting_cash=1e9, max_concurrent=10_000)
    bt = Backtest(src, strat, cfg, FillModel(slippage_frac=slip)).run()
    spot = src._spot
    out = []
    for c in bt.closed:
        m = c.position.meta
        s0, s1 = m.get("spot"), spot.get(c.exit_date)
        risk = c.position.max_loss
        if s0 is None or s1 is None or not risk or risk <= 0:
            continue
        # Position delta per share: short put contributes -delta (positive),
        # long put +delta (negative). Hedge = short that many shares x 100.
        pdelta = -float(m["short_delta"]) + float(m["long_delta"])
        hedged = c.pnl - pdelta * 100.0 * (s1 - s0)
        out.append({"sym": symbol, "entry": c.position.entry_date.isoformat(),
                    "month": c.position.entry_date.strftime("%Y-%m"),
                    "pnl": c.pnl, "hedged": hedged, "risk": risk,
                    "credit": c.position.credit_received, "pdelta": pdelta})
    return out


def _ror(rows, key):
    r = sum(x["risk"] for x in rows)
    return sum(x[key] for x in rows) / r if r else float("nan")


def _boot_t(rows, key, seed=7):
    """Return-on-risk / its block-bootstrap SE, resampling calendar months."""
    by = defaultdict(list)
    for x in rows:
        by[x["month"]].append(x)
    months = list(by)
    if len(months) < 4:
        return float("nan")
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        pick = [x for _ in months for x in by[rng.choice(months)]]
        draws.append(_ror(pick, key))
    sd = statistics.stdev(draws)
    return _ror(rows, key) / sd if sd > 0 else float("nan")


def _halves(rows, key):
    rows = sorted(rows, key=lambda x: x["entry"])
    h = len(rows) // 2
    return _ror(rows[:h], key), _ror(rows[h:], key)


def _sources(symbols):
    from run_backtest import _build_real
    out = {}
    for s in symbols:
        out[s] = _build_real(s, START, END, spread_model=SpreadModel.for_symbol(s),
                             verbose=False)
    return out


def _symbol_rows(args):
    """Every cell for one symbol, both slippages. One process per symbol: the
    reconstructed chains (the expensive IV solve) are cached per source and
    reused across all 48 cells, so a symbol's cells belong in one process."""
    symbol, cells = args
    src = _sources([symbol])[symbol]
    return symbol, {cell: {slip: _trades(src, symbol, *cell, slip) for slip in (0.5, 1.0)}
                    for cell in cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=".cache/structure_study.json")
    args = ap.parse_args()

    symbols = SYMBOLS[:1] if args.quick else SYMBOLS
    cells = [(0.20, 0.01, 7)] if args.quick else list(itertools.product(DELTAS, WIDTHS, DTES))
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(len(symbols)) as pool:
        by_sym = dict(pool.map(_symbol_rows, [(s, cells) for s in symbols]))
    print(f"backtests done in {time.time()-t0:.0f}s", flush=True)

    results = []
    for cell in cells:
        d, w, dte = cell
        rows = {slip: [r for s in symbols for r in by_sym[s][cell][slip]]
                for slip in (0.5, 1.0)}
        prim = rows[0.5]
        if len(prim) < 30:
            print(f"d={d:.2f} w={w:.1%} dte={dte:>2}: only {len(prim)} trades, skipped")
            continue
        per_sym = {s: _ror([x for x in prim if x["sym"] == s], "hedged") for s in symbols}
        h1, h2 = _halves(prim, "hedged")
        res = {
            "delta": d, "width_pct": w, "dte": dte, "n": len(prim),
            "credit": statistics.mean(x["credit"] for x in prim),
            "risk": statistics.mean(x["risk"] for x in prim),
            "raw_ror": _ror(prim, "pnl"), "raw_t": _boot_t(prim, "pnl"),
            "hedged_ror": _ror(prim, "hedged"), "hedged_t": _boot_t(prim, "hedged"),
            "per_symbol": per_sym, "half1": h1, "half2": h2,
            "hedged_ror_slip1": _ror(rows[1.0], "hedged"),
            "win_rate": sum(x["pnl"] > 0 for x in prim) / len(prim),
        }
        res["qualifies"] = bool(
            res["hedged_t"] >= T_QUALIFY
            and all(v > 0 for v in per_sym.values())
            and h1 > 0 and h2 > 0 and res["hedged_ror_slip1"] > 0)
        results.append(res)
        print(f"d={d:.2f} w={w:>4.1%} dte={dte:>2} n={len(prim):>4} "
              f"raw {res['raw_ror']:+6.2%} (t {res['raw_t']:+5.2f})  "
              f"hedged {res['hedged_ror']:+6.2%} (t {res['hedged_t']:+5.2f})  "
              f"halves {h1:+.2%}/{h2:+.2%}  slip1 {res['hedged_ror_slip1']:+.2%}  "
              f"{'QUALIFIES' if res['qualifies'] else ''}", flush=True)

    if not args.quick:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        q = [r for r in results if r["qualifies"]]
        print(f"\n{len(q)} of {len(results)} cells qualify.")
        best = max(results, key=lambda r: r["hedged_t"], default=None)
        if best:
            print(f"highest hedged t: d={best['delta']} w={best['width_pct']:.1%} "
                  f"dte={best['dte']} t={best['hedged_t']:+.2f} -- expected max of "
                  f"{len(results)} noise draws is ~{math.sqrt(2*math.log(len(results))):.1f}")


if __name__ == "__main__":
    sys.exit(main())
