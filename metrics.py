"""Turn a finished Backtest into an honest scorecard.

Pure Python, no numpy dependency. The numbers that matter for a premium-selling
strategy are NOT win rate (which is misleadingly high by design) -- they are
expectancy per trade and max drawdown, because the whole risk lives in the tail.
"""
from __future__ import annotations

from statistics import mean


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def max_drawdown(equity_curve):
    peak, mdd, peak_eq = -float("inf"), 0.0, None
    for _, eq in equity_curve:
        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak else 0.0
        if dd < mdd:
            mdd, peak_eq = dd, peak
    return mdd  # negative fraction, e.g. -0.18 = -18%


def summarize(bt) -> dict:
    trades = bt.closed
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    start = bt.cfg.starting_cash
    total = sum(pnls)
    s = sorted(pnls)

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "total_pnl": total,
        "return_pct": total / start if start else 0.0,
        "expectancy": mean(pnls) if pnls else 0.0,
        "avg_win": mean(wins) if wins else 0.0,
        "avg_loss": mean(losses) if losses else 0.0,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses else float("inf"),
        "max_drawdown": max_drawdown(bt.equity_curve),
        "avg_days_held": mean([t.days_held for t in trades]) if trades else 0.0,
        "worst_trade": min(pnls) if pnls else 0.0,
        "best_trade": max(pnls) if pnls else 0.0,
        "pnl_p05": _percentile(s, 0.05),
        "pnl_p50": _percentile(s, 0.50),
        "pnl_p95": _percentile(s, 0.95),
    }


def print_report(bt, title=""):
    m = summarize(bt)
    exit_reasons = {}
    for t in bt.closed:
        exit_reasons[t.reason] = exit_reasons.get(t.reason, 0) + 1

    print("=" * 60)
    if title:
        print(title)
        print("-" * 60)
    print(f"Trades              {m['trades']}")
    print(f"Win rate            {m['win_rate']*100:5.1f}%")
    print(f"Total P&L           ${m['total_pnl']:,.0f}")
    print(f"Return on capital   {m['return_pct']*100:5.1f}%")
    print(f"Expectancy / trade  ${m['expectancy']:,.1f}")
    print(f"Avg win / avg loss  ${m['avg_win']:,.1f} / ${m['avg_loss']:,.1f}")
    print(f"Profit factor       {m['profit_factor']:.2f}")
    print(f"Max drawdown        {m['max_drawdown']*100:5.1f}%")
    print(f"Avg days held       {m['avg_days_held']:.1f}")
    print(f"Worst / best trade  ${m['worst_trade']:,.0f} / ${m['best_trade']:,.0f}")
    print(f"P&L p05/p50/p95     ${m['pnl_p05']:,.0f} / ${m['pnl_p50']:,.0f} / ${m['pnl_p95']:,.0f}")
    print(f"Exit reasons        {exit_reasons}")
    print("=" * 60)
    return m
