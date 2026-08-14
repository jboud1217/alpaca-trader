"""ENGINE VALIDATION against a published external benchmark.

The CBOE PUT index sells ATM SPX puts monthly, fully collateralised in T-bills,
held to expiry. That is a real, published, daily-marked implementation of the
same trade this repo backtests. If my engine cannot land near it on a matched
configuration, my engine is wrong and every result in this repo is suspect.

Approximation, stated plainly: SPY not SPX, 50-delta not exactly ATM, and a
60-wide long wing instead of true cash-secured (the engine only prices defined
risk). The wing is far enough out to be nearly worthless, so the payoff is
close to naked, but it does cap the tail the real index bears."""
import csv, os, statistics
from datetime import date, datetime
from data import AlpacaDataSource
from engine import Backtest, BacktestConfig, FillModel
from spread import SpreadModel
from strategies import PutCreditSpread

S, E = date(2025,2,10), date(2026,8,7)
YEARS = (E - S).days / 365.25

# ---- the benchmark -------------------------------------------------------
rows = []
with open("/private/tmp/claude-501/-Users-jboud1217-repos-alpaca-trading/ef169eaf-5a6d-48d5-8030-375c7a28acf3/scratchpad/PUT_History.csv") as fh:
    for r in csv.DictReader(fh):
        try:
            d = datetime.strptime(r["DATE"].strip(), "%m/%d/%Y").date()
            rows.append((d, float(r["PUT"])))
        except (ValueError, KeyError):
            continue
rows.sort()
win = [(d, v) for d, v in rows if S <= d <= E]
put_ret = win[-1][1]/win[0][1] - 1
print(f"CBOE PUT index  {win[0][0]} -> {win[-1][0]}   {win[0][1]:.2f} -> {win[-1][1]:.2f}")
print(f"  total return  {put_ret:+.2%}   annualised {(1+put_ret)**(1/YEARS)-1:+.2%}")
print(f"  (includes T-bill collateral income; the option leg alone is less)\n")

# ---- my engine, matched as closely as it can be --------------------------
src = AlpacaDataSource(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"],
                       max_dte=75, strike_band=0.12,
                       spread_model=SpreadModel.spy_measured(), verbose=False)
src.prepare("SPY", S, E)
CASH = 100_000.0
bt = Backtest(src, PutCreditSpread(short_delta=0.50, wing_width=60, target_dte=30,
              profit_take=None, stop_mult=None, min_dte=0),
              BacktestConfig(underlying="SPY", start=S, end=E,
                             starting_cash=CASH, max_concurrent=1),
              FillModel(slippage_frac=0.5)).run()
p = [t.pnl for t in bt.closed]
# COLLATERAL, done properly. The PUT index is CASH-SECURED: it sets aside the
# full strike x 100 per contract, because a naked put's loss runs all the way to
# zero. Scaling by the 60-point wing instead ($6,000) silently ran ~13x the
# index's leverage and produced a 6x return discrepancy. Match the risk, not
# the structure.
strikes = [t.position.meta.get("short_strike") for t in bt.closed
           if t.position.meta.get("short_strike")]
collateral = (sum(strikes)/len(strikes)) * 100 if strikes else 775 * 100
n_units = CASH / collateral
eng_ret = sum(p) * n_units / CASH
print(f"engine, SPY 50-delta / 30 DTE / 60-wide / held to expiry")
print(f"  trades {len(p)}   mean {statistics.mean(p):+.2f}   total ${sum(p):+,.0f} per 1 contract")
print(f"  scaled to full collateralisation ({n_units:.1f} contracts): {eng_ret:+.2%}")
print(f"  + T-bill on collateral @4.2%: {eng_ret + 0.042*YEARS:+.2%}")
print(f"  annualised {((1+eng_ret+0.042*YEARS)**(1/YEARS)-1):+.2%}")
print()
eng_ann = (1+eng_ret+0.042*YEARS)**(1/YEARS)-1
put_ann = (1+put_ret)**(1/YEARS)-1
gap = eng_ann - put_ann
print(f"avg short strike ${collateral/100:,.0f} -> {n_units:.2f} contracts on ${CASH:,.0f}")
print(f"GAP vs benchmark: {gap:+.2%}/yr")
verdict = ("PASS - within 2%/yr" if abs(gap) < 0.02 else
           "MARGINAL - within 5%/yr" if abs(gap) < 0.05 else
           "FAIL - engine disagrees with the benchmark materially")
print(f"VERDICT: {verdict}")
