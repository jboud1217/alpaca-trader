"""Entry point.

Default run is fully offline: it backtests both templates against the synthetic
data source so you can see the machine work end to end with no keys or network.

    python run_backtest.py                        # both templates, synthetic
    python run_backtest.py --sweep                # grid-search PCS params
    python run_backtest.py --calibrate -s SPY     # fit spread model to a live chain
    python run_backtest.py --real -s SPY          # backtest on real Alpaca history
    python run_backtest.py --real -s SPY --band   # ...across fill assumptions

Remember what a synthetic result means: the synthetic data has a variance-risk-
premium edge baked in, so premium selling "working" there validates the ENGINE,
not the trade.

And remember what a REAL result means here, which is less than you would like.
Alpaca serves no historical option quotes, so --real observes daily trade bars
and models the bid/ask around them. Every real-data P&L is conditional on that
model. This is why --band exists and why you should read it instead of the
single-run number: if the sign of the edge flips across the band, you do not
have an edge, you have a fill assumption. Not financial advice.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from datetime import date, timedelta

from data import SyntheticDataSource
from engine import Backtest, BacktestConfig, FillModel
from metrics import print_report, summarize
from strategies import IronCondor, PutCreditSpread


# --------------------------------------------------------------------------- #
# Synthetic (offline)
# --------------------------------------------------------------------------- #
def build_synthetic():
    return SyntheticDataSource(
        start=date(2023, 1, 1),
        end=date(2025, 12, 31),
        s0=100.0,
        realized_vol=0.18,
        vrp=0.03,          # the edge the engine should be able to harvest
        drift=0.06,
        seed=11,
    )


def templates():
    return (
        PutCreditSpread(short_delta=0.30, wing_width=5, target_dte=45,
                        profit_take=0.50, stop_mult=2.0, min_dte=21),
        IronCondor(short_delta=0.20, wing_width=5, target_dte=45,
                   profit_take=0.50, stop_mult=2.0, min_dte=21),
    )


def single_run():
    data = build_synthetic()
    cfg = BacktestConfig(underlying="SYN", start=date(2023, 1, 1),
                         end=date(2025, 12, 31), starting_cash=25_000, max_concurrent=1)
    fills = FillModel(slippage_frac=0.5)
    for strat in templates():
        bt = Backtest(data, strat, cfg, fills).run()
        print_report(bt, title=f"{strat.name}  (synthetic, slippage_frac=0.5)")


def sweep():
    data = build_synthetic()
    cfg = BacktestConfig(underlying="SYN", start=date(2023, 1, 1),
                         end=date(2025, 12, 31), starting_cash=25_000, max_concurrent=1)
    fills = FillModel(slippage_frac=0.5)

    print(f"{'delta':>6} {'dte':>4} {'take':>5} | {'trades':>7} {'win%':>6} "
          f"{'exp$':>8} {'maxDD%':>7} {'PF':>5}")
    print("-" * 60)
    for d, dte, take in itertools.product([0.20, 0.30, 0.40], [30, 45], [0.50, 0.75]):
        strat = PutCreditSpread(short_delta=d, wing_width=5, target_dte=dte,
                                profit_take=take, stop_mult=2.0, min_dte=21)
        bt = Backtest(data, strat, cfg, fills).run()
        m = summarize(bt)
        print(f"{d:>6.2f} {dte:>4} {take:>5.2f} | {m['trades']:>7} "
              f"{m['win_rate']*100:>5.1f} {m['expectancy']:>8.1f} "
              f"{m['max_drawdown']*100:>6.1f} {m['profit_factor']:>5.2f}")


# --------------------------------------------------------------------------- #
# Real Alpaca history
# --------------------------------------------------------------------------- #
def _keys():
    """Alpaca's own env var names first, then the friendlier aliases."""
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit(
            "Missing credentials. Export a PAPER key pair (read-only market data is\n"
            "all this needs -- it places no orders):\n\n"
            "    export APCA_API_KEY_ID=...\n"
            "    export APCA_API_SECRET_KEY=...\n"
        )
    return key, sec


def _build_real(symbol, start, end, spread_model=None, verbose=True,
                strike_band=0.10):
    from data import AlpacaDataSource
    key, sec = _keys()
    # strike_band drives the whole download cost. 0.10 is ample for these
    # templates -- a 30-delta put at 45 DTE sits ~5% OTM and the wings are $5 --
    # and on a name with daily expirations like SPY a wider band multiplies the
    # contract universe into six figures for strikes nothing ever trades.
    src = AlpacaDataSource(key, sec, max_dte=60, strike_band=strike_band,
                           spread_model=spread_model, verbose=verbose)
    src.prepare(symbol, start, end)
    return src


def calibrate(symbol):
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from spread import calibrate_from_live_chain

    key, sec = _keys()
    client = OptionHistoricalDataClient(key, sec)
    res = calibrate_from_live_chain(client, symbol)
    m = res["model"]
    print(f"\nSpread calibration for {symbol}")
    print(f"  {res['n_traded_band']} contracts in the traded band "
          f"(|delta| 0.10-0.45, 15-60 DTE) out of {res['n_all_quoted']} quoted")
    print("-" * 66)
    p25, p50, p75 = res["half_spread_p25_p50_p75"]
    print(f"  half-spread, traded band   p25/p50/p75   ${p25:.3f} / ${p50:.3f} / ${p75:.3f}")
    p25, p50, p75 = res["half_frac_p25_p50_p75"]
    print(f"  as fraction of mid         p25/p50/p75   {p25:.1%} / {p50:.1%} / {p75:.1%}")
    p25, p50, p75 = res["all_half_frac_p25_p50_p75"]
    print(f"  ...across the WHOLE chain  p25/p50/p75   {p25:.1%} / {p50:.1%} / {p75:.1%}"
          f"   <- why the filter matters")
    print(f"\n  fitted: SpreadModel(min_half={m.min_half}, pct_of_price={m.pct_of_price})")
    print(f"\n  {res['note']}")


def real_run(symbol, start, end, band=False, strike_band=0.10):
    from spread import SpreadModel

    cfg = BacktestConfig(underlying=symbol, start=start, end=end,
                         starting_cash=25_000, max_concurrent=1)
    src = _build_real(symbol, start, end, strike_band=strike_band)
    days = src.trading_days(start, end)
    print(f"[alpaca] trading days in window: {len(days)}")

    if not band:
        for strat in templates():
            bt = Backtest(src, strat, cfg, FillModel(slippage_frac=0.5)).run()
            print_report(bt, title=f"{strat.name}  ({symbol}, REAL, "
                                   f"modelled spread, slippage_frac=0.5)")
        print(f"\n[alpaca] contracts skipped: {src.skipped}")
        print("[alpaca] Single-run numbers are conditional on the spread model. "
              "Re-run with --band.")
        return

    # The honest presentation: total P&L across the assumption grid, because the
    # assumptions are the dominant source of variance in the result.
    spreads = [("tight (=SPY)", SpreadModel.tight()),
               ("optimistic", SpreadModel.optimistic()),
               ("realistic", SpreadModel.realistic()),
               ("pessimistic", SpreadModel.pessimistic())]
    slips = [0.0, 0.5, 1.0]

    for strat_factory in (
        lambda: PutCreditSpread(short_delta=0.30, wing_width=5, target_dte=45,
                                profit_take=0.50, stop_mult=2.0, min_dte=21),
        lambda: IronCondor(short_delta=0.20, wing_width=5, target_dte=45,
                           profit_take=0.50, stop_mult=2.0, min_dte=21),
    ):
        name = strat_factory().name
        print(f"\n{'='*66}\n{name}  --  {symbol} real history {start}..{end}")
        print("total P&L on $25k, by fill assumption")
        print(f"{'':>14}" + "".join(f"{'slip='+str(s):>16}" for s in slips))
        print("-" * 66)
        for label, sm in spreads:
            src.spread_model = sm
            row = f"{label:>14}"
            for slip in slips:
                bt = Backtest(src, strat_factory(), cfg, FillModel(slippage_frac=slip)).run()
                m = summarize(bt)
                tot = sum(t.pnl for t in bt.closed)
                row += f"{tot:>10.0f} ({m['trades']:>3})"
            print(row)
        print("\n(cell = total P&L, (n) = trades)")

    print(f"\n[alpaca] contracts skipped: {src.skipped}")
    print("\nHow to read this: if the sign flips anywhere in the grid, the result is\n"
          "a fill assumption rather than an edge. Only a table that stays positive\n"
          "in the pessimistic/slip=1.0 corner is worth paper-trading.")


def friction_sweep(symbol, start, end, strike_band=0.10):
    """Attack the cost side rather than hunt for edge.

    Measured on the SPY run: round-trip spread was ~13% of credit against an
    expectancy of ~15% of credit. Friction is nearly the entire result, and
    unlike edge it can be reduced by construction rather than by discovery --
    no statistics required to believe a cheaper structure is cheaper.

    Three levers, swept together because they interact:
      width    credit scales with wing width; friction scales with LEG COUNT,
               which is 2 either way. Wider should mean less friction per
               dollar of credit.
      expire   letting a far-OTM position expire concedes no closing spread at
               all, versus buying it back and paying the full half-spread.
      take     a 50% profit target forces a round trip on every winner. Not
               taking it trades more decay for fewer fills.

    Results are reported per dollar AT RISK, not raw, because a wider spread
    risks proportionally more -- a raw-P&L comparison across widths would just
    reward taking more risk and call it improvement.
    """
    import math
    import statistics
    from spread import SpreadModel

    cfg = BacktestConfig(underlying=symbol, start=start, end=end,
                         starting_cash=25_000, max_concurrent=1)
    src = _build_real(symbol, start, end, spread_model=SpreadModel.tight(),
                      strike_band=strike_band)
    fills = FillModel(slippage_frac=0.5)

    print(f"\n{'='*104}")
    print(f"friction sweep -- {symbol} {start}..{end}, measured (tight) spread, slip=0.5")
    print(f"{'='*104}")
    print(f"{'width':>6}{'expire':>8}{'take':>6} | {'trades':>7}{'totP&L':>9}{'exp$':>8}"
          f"{'credit':>8} | {'fric%':>7}{'open':>7}{'close':>7} | {'exp/risk':>9}{'t':>6}"
          f"  exits")
    print("-" * 104)

    rows = []
    for width in (5, 10, 20, 30):
        for expire in (None, 0.10):
            for take in (0.50, None):
                strat = PutCreditSpread(
                    short_delta=0.30, wing_width=width, target_dte=45,
                    profit_take=take, stop_mult=2.0, min_dte=21,
                    expire_below_delta=expire)
                bt = Backtest(src, strat, cfg, fills).run()
                if not bt.closed:
                    continue
                pnls = [t.pnl for t in bt.closed]
                n = len(pnls)
                mean = statistics.mean(pnls)
                sd = statistics.stdev(pnls) if n > 2 else 0.0
                t_stat = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
                risks = [p.position.max_loss for p in bt.closed
                         if p.position.max_loss]
                avg_risk = statistics.mean(risks) if risks else float("nan")
                exits = {}
                for c in bt.closed:
                    k = c.reason.split()[0]
                    exits[k] = exits.get(k, 0) + 1
                top = ",".join(f"{k}:{v}" for k, v in
                               sorted(exits.items(), key=lambda x: -x[1])[:3])
                rows.append((width, expire, take, n, sum(pnls), mean,
                             bt.friction_frac, mean / avg_risk, t_stat))
                print(f"{width:>6}{('%.2f' % expire) if expire else '--':>8}"
                      f"{('%.2f' % take) if take else '--':>6} | {n:>7}{sum(pnls):>9.0f}"
                      f"{mean:>8.1f}{bt.credit_collected/max(n,1):>8.0f} | "
                      f"{bt.friction_frac:>6.1%}{bt.spread_cost_open/max(bt.credit_collected,1):>6.1%}"
                      f"{bt.spread_cost_close/max(bt.credit_collected,1):>6.1%} | "
                      f"{mean/avg_risk:>8.2%}{t_stat:>6.2f}  {top}")

    print("-" * 104)
    if rows:
        base = next((r for r in rows if r[0] == 5 and r[1] is None and r[2] == 0.50), None)
        best = max(rows, key=lambda r: r[7])
        if base:
            print(f"baseline (5-wide, close at 50%): friction {base[6]:.1%} of credit, "
                  f"exp/risk {base[7]:.2%}, t={base[8]:.2f}")
        print(f"lowest-friction-adjusted:        width={best[0]} "
              f"expire={best[1]} take={best[2]}: friction {best[6]:.1%}, "
              f"exp/risk {best[7]:.2%}, t={best[8]:.2f}")
        print("\nexp/risk is expectancy per dollar of max loss -- the only fair "
              "comparison across widths.\nA better t here is still not "
              "significance; 16 configs on ~40 trades each will produce a best\n"
              "cell by chance alone. Read the FRICTION columns, which are "
              "arithmetic, not inference.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="grid-search parameters (synthetic)")
    ap.add_argument("--real", action="store_true", help="backtest on real Alpaca history")
    ap.add_argument("--band", action="store_true", help="with --real: sweep fill assumptions")
    ap.add_argument("--calibrate", action="store_true", help="fit spread model to a live chain")
    ap.add_argument("--friction", action="store_true",
                    help="sweep width / expire-worthless / profit-take against fill cost")
    ap.add_argument("-s", "--symbol", default="SPY", help="underlying (default SPY)")
    ap.add_argument("--start", type=date.fromisoformat, default=None,
                    help="YYYY-MM-DD (default: 18 months back, floored at 2024-02-01)")
    ap.add_argument("--end", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
    ap.add_argument("--strike-band", type=float, default=0.10,
                    help="fraction of spot to keep either side (drives download size)")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.symbol)
    elif args.friction:
        end = args.end or date.today() - timedelta(days=1)
        start = args.start or max(date(2024, 2, 1), end - timedelta(days=548))
        friction_sweep(args.symbol, start, end, strike_band=args.strike_band)
    elif args.real:
        end = args.end or date.today() - timedelta(days=1)
        start = args.start or max(date(2024, 2, 1), end - timedelta(days=548))
        real_run(args.symbol, start, end, band=args.band, strike_band=args.strike_band)
    elif args.sweep:
        sweep()
    else:
        single_run()
