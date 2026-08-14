"""Live event scanner: poll -> score -> suggest. Never trades.

    python scanner.py --weights                 # print the weight table + provenance
    python scanner.py --once -s SPY,QQQ,IWM     # one pass, print and log
    python scanner.py --watch --interval 300    # poll every 5 min during RTH
    python scanner.py --review                  # score past suggestions vs outcomes

WHAT THIS DOES NOT DO
---------------------
It places no orders. It imports no order-submission API. `TradingClient` is used
only for the market clock and the calendar. Output is a ranked list of
candidates with the arithmetic attached, written to a log; acting on any of them
is a manual decision you make with your own eyes on the chain.

That is not timidity, it is sequencing. Per weights.py, most of the table is
PRIOR -- reasoned guesses, not measured effects -- and the underlying templates
have not yet cleared realistic fill costs even on data with an edge baked in. An
automated order path on top of that stack would be automating something nobody
has shown works yet.

THE LOG IS THE POINT
--------------------
Every suggestion is appended to suggestions.jsonl with its full factor
breakdown, the live quotes at the time, and an empty outcome slot. `--review`
fills those slots in from subsequent price history. That is the mechanism by
which PRIOR weights can eventually become MEASURED ones. Without the log the
scanner is a random-number generator with good manners.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import events as ev
import weights as W
from strategies import IronCondor, PutCreditSpread

SUGGESTION_LOG = Path("suggestions.jsonl")
IV_HISTORY_DIR = Path(".cache/iv_history")


# --------------------------------------------------------------------------- #
def _clients():
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY.")
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    return {
        "option": OptionHistoricalDataClient(key, sec),
        "stock": StockHistoricalDataClient(key, sec),
        "news": NewsClient(key, sec),
        "ca": CorporateActionsClient(key, sec),
        # clock/calendar only -- no order methods are called anywhere in this file
        "trading": TradingClient(key, sec, paper=True),
    }


# --------------------------------------------------------------------------- #
# IV history (bootstraps the iv_rank factor)
# --------------------------------------------------------------------------- #
def _iv_history_path(symbol: str) -> Path:
    IV_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return IV_HISTORY_DIR / f"{symbol.upper()}.json"


def record_iv(symbol: str, iv: float, on: Optional[date] = None) -> List[float]:
    """Append today's ATM IV (one observation per calendar day) and return the
    full series. This is how iv_rank becomes available: it simply does not
    report until the scanner has watched the name for ~3 months."""
    on = on or date.today()
    p = _iv_history_path(symbol)
    hist = json.loads(p.read_text()) if p.exists() else {}
    hist[on.isoformat()] = iv
    p.write_text(json.dumps(hist, sort_keys=True))
    return [hist[k] for k in sorted(hist)]


def iv_percentile(series: List[float], current: float) -> Optional[float]:
    if len(series) < 60:
        return None
    below = sum(1 for x in series if x <= current)
    return below / len(series)


# --------------------------------------------------------------------------- #
def evaluate(symbol: str, clients: dict, earnings: ev.EarningsCalendar, *,
             target_dte: int = 3, max_dte: int = 60,
             require_earnings_data: bool = True) -> Optional[dict]:
    """Gather events for one symbol, score them, and propose a concrete trade."""
    symbol = symbol.upper()
    vetoes: List[str] = []

    stats = ev.underlying_stats(clients["stock"], symbol)
    if stats is None:
        return {"symbol": symbol, "error": "no underlying price history"}

    chain = ev.live_chain(clients["option"], symbol, stats.spot, max_dte=max_dte)
    if chain is None:
        return {"symbol": symbol, "error": "no live two-sided option market"}

    # ---- events -------------------------------------------------------- #
    iv_atm = ev.atm_iv(chain, 30)
    slope = ev.term_structure_slope(chain)
    burst = ev.news_burst(clients["news"], symbol)
    exdiv, exdiv_src = ev.next_ex_dividend(clients["ca"], symbol)

    expiry = chain.nearest_expiry(target_dte)
    if expiry is None:
        return {"symbol": symbol, "error": "no expiry near target dte"}

    # ---- earnings: a veto, not a weight -------------------------------- #
    nxt = None if symbol in ev.NO_EARNINGS_SYMBOLS else earnings.next_earnings(symbol)
    if nxt == ev.UNKNOWN:
        if require_earnings_data:
            vetoes.append("earnings date UNKNOWN (no calendar entry; not assuming none)")
    elif isinstance(nxt, date) and nxt <= expiry:
        vetoes.append(f"earnings {nxt} falls on/before expiry {expiry}")

    # ---- propose a concrete structure ---------------------------------- #
    strat = PutCreditSpread(short_delta=0.30, wing_width=3, target_dte=target_dte,
                            profit_take=0.50, stop_mult=2.0, min_dte=1)
    pos = strat.propose_entry(chain, 0)
    if pos is None:
        return {"symbol": symbol, "error": "no strike pair matched the template"}

    short_leg = max(pos.legs, key=lambda l: l.strike)
    long_leg = min(pos.legs, key=lambda l: l.strike)
    sq = chain.find("put", short_leg.strike, expiry)
    lq = chain.find("put", long_leg.strike, expiry)
    credit = (sq.bid - lq.ask) * 100 if sq and lq else None
    half_spread = (0.5 * (sq.ask - sq.bid) + 0.5 * (lq.ask - lq.bid)) * 100 if sq and lq else None
    width = (short_leg.strike - long_leg.strike) * 100
    if credit is None or credit <= 0:
        vetoes.append("template produces no net credit at live quotes")
    else:
        sv = W.spread_veto(half_spread, credit)
        if sv:
            vetoes.append(sv)

    # Dividend-driven early assignment threatens SHORT CALLS: the holder of a
    # deep-ITM call exercises the day before ex-div to capture the dividend when
    # the call's remaining extrinsic is worth less than the payout. A short PUT
    # has the opposite exposure -- a dividend makes puts more valuable to hold,
    # not less -- so vetoing put spreads on ex-div blocks trades for a risk that
    # is not present. Gate on the structure actually having a short call.
    has_short_call = any(l.action == "sell" and l.right == "call" for l in pos.legs)
    if has_short_call and exdiv and exdiv <= expiry:
        vetoes.append(f"ex-dividend {exdiv} ({exdiv_src}) before expiry "
                      "(early-assignment risk on the short call)")

    # ---- score ---------------------------------------------------------- #
    series = record_iv(symbol, iv_atm) if iv_atm else []
    pctile = iv_percentile(series, iv_atm) if iv_atm else None

    values = {
        "iv_minus_rv":    W.norm_iv_minus_rv(iv_atm, stats.realized_vol_20d),
        "iv_rank":        W.norm_iv_rank(pctile, len(series)),
        "term_slope":     W.norm_term_slope(slope),
        "trend_stress":   W.norm_trend_stress(stats.drawdown_20d, stats.pct_from_20d_ma),
        "spread_quality": W.norm_spread_quality(half_spread, credit),
        "news_intensity": W.norm_news_intensity(burst.intensity if burst else None),
    }
    sc = W.score(values, vetoes)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "spot": round(stats.spot, 2),
        "composite": round(sc.composite, 4),
        "coverage": round(sc.coverage, 3),
        "vetoes": sc.vetoes,
        "structure": {
            "type": "put_credit_spread",
            "expiry": expiry.isoformat(),
            "dte": (expiry - chain.as_of).days,
            "short_strike": short_leg.strike,
            "long_strike": long_leg.strike,
            "credit": round(credit, 2) if credit else None,
            "width": width,
            "max_loss": round(width - credit, 2) if credit else None,
            "half_spread_cost": round(half_spread, 2) if half_spread else None,
            "short_delta": round(sq.delta, 3) if sq else None,
        },
        "events": {
            "atm_iv_30d": round(iv_atm, 4) if iv_atm else None,
            "realized_vol_20d": round(stats.realized_vol_20d, 4) if stats.realized_vol_20d else None,
            "term_slope": round(slope, 4) if slope else None,
            "iv_obs_count": len(series),
            "news_24h": burst.count_24h if burst else None,
            "news_intensity": round(burst.intensity, 2) if burst else None,
            "headlines": burst.latest_headlines if burst else [],
            "next_ex_div": exdiv.isoformat() if exdiv else None,
            "next_ex_div_source": exdiv_src,
            "next_earnings": (nxt.isoformat() if isinstance(nxt, date)
                              else (nxt if isinstance(nxt, str) else None)),
            "drawdown_20d": round(stats.drawdown_20d, 4) if stats.drawdown_20d else None,
        },
        "factors": [
            {"key": f.key, "value": f.value, "weight": f.weight,
             "provenance": f.provenance.value, "detail": f.detail}
            for f in sc.factors
        ],
        "outcome": None,        # filled by --review
        "_explain": sc.explain(),
    }


# --------------------------------------------------------------------------- #
def log_suggestion(rec: dict) -> None:
    rec = {k: v for k, v in rec.items() if k != "_explain"}
    with SUGGESTION_LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_once(symbols: List[str], clients: dict, earnings: ev.EarningsCalendar,
             threshold: float, min_coverage: float, quiet: bool = False) -> List[dict]:
    results = []
    for sym in symbols:
        try:
            rec = evaluate(sym, clients, earnings)
        except Exception as e:
            rec = {"symbol": sym, "error": f"{type(e).__name__}: {e}"}
        results.append(rec)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*74}\nscan {stamp}")
    ranked = [r for r in results if "composite" in r]
    ranked.sort(key=lambda r: (-1e9 if r["vetoes"] else 0, -r["composite"]))

    for r in results:
        if "error" in r:
            print(f"\n{r['symbol']:<6} -- skipped: {r['error']}")
    for r in ranked:
        s = r["structure"]
        flag = "VETOED" if r["vetoes"] else ("SUGGEST" if
               (r["composite"] >= threshold and r["coverage"] >= min_coverage) else "watch")
        print(f"\n{r['symbol']:<6} {flag:<8} composite {r['composite']:+.3f}  "
              f"coverage {r['coverage']:.0%}  spot {r['spot']}")
        if s["credit"]:
            print(f"       {s['short_strike']:.0f}/{s['long_strike']:.0f}p  {s['expiry']} "
                  f"({s['dte']}d)  credit ${s['credit']:.0f}  max loss ${s['max_loss']:.0f}  "
                  f"delta {s['short_delta']}")
            print(f"       round-trip spread cost ${s['half_spread_cost']:.0f} "
                  f"= {s['half_spread_cost']/s['credit']:.0%} of credit")
        if not quiet:
            print(r["_explain"])
        log_suggestion(r)

    if ranked:
        n_sug = sum(1 for r in ranked if not r["vetoes"]
                    and r["composite"] >= threshold and r["coverage"] >= min_coverage)
        print(f"\n{n_sug} of {len(ranked)} above threshold {threshold:+.2f}. "
              f"Logged to {SUGGESTION_LOG}. No orders placed.")
    return results


def market_open(clients: dict) -> bool:
    try:
        return bool(clients["trading"].get_clock().is_open)
    except Exception:
        return True     # fail open: a clock outage should not silently stop the scan


def watch(symbols, clients, earnings, threshold, min_coverage, interval, rth_only):
    print(f"scanner: {len(symbols)} symbols, every {interval}s"
          f"{' during RTH only' if rth_only else ''}. Ctrl-C to stop.")
    while True:
        try:
            if rth_only and not market_open(clients):
                print(f"[{datetime.now():%H:%M:%S}] market closed, sleeping", flush=True)
            else:
                run_once(symbols, clients, earnings, threshold, min_coverage, quiet=True)
        except KeyboardInterrupt:
            print("\nstopped."); return
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] scan error: {type(e).__name__}: {e}", flush=True)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nstopped."); return


# --------------------------------------------------------------------------- #
def review(clients: dict, horizon_days: int = 21) -> None:
    """Score past suggestions against what the underlying actually did.

    Crude by design -- it marks whether the short strike was breached within the
    horizon, not the actual P&L of a managed position. That is enough to tell
    whether high-composite suggestions breach less often than low ones, which is
    the question that decides whether the weights mean anything.
    """
    if not SUGGESTION_LOG.exists():
        print(f"No {SUGGESTION_LOG} yet. Run --once or --watch first."); return

    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    recs = [json.loads(l) for l in SUGGESTION_LOG.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "composite" in r and not r["vetoes"]]
    if not recs:
        print("No un-vetoed suggestions logged yet."); return

    buckets: Dict[str, List[int]] = {}
    scored = 0
    for r in recs:
        ts = datetime.fromisoformat(r["ts"])
        start = ts.date()
        end = min(start + timedelta(days=horizon_days), date.today() - timedelta(days=1))
        if end <= start:
            continue        # too recent to judge
        try:
            bars = clients["stock"].get_stock_bars(StockBarsRequest(
                symbol_or_symbols=r["symbol"], timeframe=TimeFrame.Day,
                start=datetime.combine(start, datetime.min.time()),
                end=datetime.combine(end, datetime.max.time()),
                adjustment=Adjustment.RAW)).data.get(r["symbol"], [])
        except Exception:
            continue
        if not bars:
            continue
        low = min(float(b.low) for b in bars)
        breached = int(low <= r["structure"]["short_strike"])
        b = ("high  (>= +0.20)" if r["composite"] >= 0.20 else
             "mid   (0..0.20)" if r["composite"] >= 0 else "low   (< 0)")
        buckets.setdefault(b, []).append(breached)
        scored += 1

    print(f"\nReviewed {scored} suggestions with >= {horizon_days}d of history")
    print("-" * 58)
    print(f"{'composite bucket':<20}{'n':>5}{'breach rate':>14}")
    for b in sorted(buckets):
        v = buckets[b]
        print(f"{b:<20}{len(v):>5}{sum(v)/len(v):>13.0%}")
    print("\nIf breach rate does not fall as composite rises, the weights are not\n"
          "earning their keep. That is the signal to change them -- or drop them.")
    if scored < 30:
        print(f"\n{scored} observations is far too few to conclude anything. "
              "This needs months.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--weights", action="store_true", help="print weight table + provenance")
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls")
    ap.add_argument("--threshold", type=float, default=0.15, help="min composite to suggest")
    ap.add_argument("--min-coverage", type=float, default=0.60,
                    help="min fraction of weight computable before trusting a score")
    ap.add_argument("--earnings-csv", default="earnings.csv")
    ap.add_argument("--all-hours", action="store_true", help="poll outside RTH too")
    args = ap.parse_args()

    if args.weights:
        print(W.provenance_report()); sys.exit(0)

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cl = _clients()
    cal = ev.EarningsCalendar(args.earnings_csv)
    if not cal.loaded:
        missing = [s for s in syms if s not in ev.NO_EARNINGS_SYMBOLS]
        if missing:
            print(f"NOTE: no {args.earnings_csv}; {', '.join(missing)} will be VETOED on "
                  f"unknown earnings. Index/ETF symbols are exempt.")

    if args.review:
        review(cl)
    elif args.watch:
        watch(syms, cl, cal, args.threshold, args.min_coverage,
              args.interval, rth_only=not args.all_hours)
    else:
        run_once(syms, cl, cal, args.threshold, args.min_coverage)
