"""The approval loop: scan -> size -> text -> await reply -> re-price -> submit.

    python alerts.py --dry-run                 # console output, no SMS, no orders
    python alerts.py --sms                     # real texts, still no orders
    python alerts.py --sms --arm               # texts, confirmed orders to PAPER
    python alerts.py --sms --arm --live-money  # ...to the LIVE account

`--arm` and `--live-money` are separate on purpose. Arming enables submission;
`--live-money` chooses whose money. Neither is implied by the other and neither
is the default, so no single mistyped flag can point real capital at this.

Default target is the paper account even when armed.

State lives in pending.json so a launchd job that fires every few minutes can
pick up a reply sent between runs. Proposals expire on the risk-limit TTL --
an approval that arrives after the quote has gone stale is not consent.
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
from typing import Dict, List

import events as ev
import execution as ex
import notify as nt
import weights as W
from strategies import PutCreditSpread

PENDING = Path("pending.json")


# --------------------------------------------------------------------------- #
def _load_pending() -> Dict[str, dict]:
    if not PENDING.exists():
        return {}
    try:
        return json.loads(PENDING.read_text())
    except json.JSONDecodeError:
        return {}


def _save_pending(p: Dict[str, dict]) -> None:
    PENDING.write_text(json.dumps(p, indent=2, default=str))


def _to_trade(d: dict) -> ex.SizedTrade:
    d = dict(d)
    d["expiry"] = date.fromisoformat(d["expiry"]) if isinstance(d["expiry"], str) else d["expiry"]
    return ex.SizedTrade(**d)


def _expire_stale(pending: Dict[str, dict], limits: ex.RiskLimits) -> List[str]:
    now = datetime.now(timezone.utc)
    dropped = []
    for tok, rec in list(pending.items()):
        age = (now - datetime.fromisoformat(rec["created_at"])).total_seconds()
        if age > limits.approval_ttl_seconds:
            dropped.append(tok)
            del pending[tok]
    return dropped


# --------------------------------------------------------------------------- #
def scan_and_propose(symbols, clients, cal, limits, notifier, executor,
                     threshold, min_coverage, pending) -> None:
    for sym in symbols:
        if any(r["underlying"] == sym for r in pending.values()):
            continue                       # already one outstanding on this name
        try:
            rec = _evaluate(sym, clients, cal)
        except Exception as e:
            print(f"[{sym}] evaluate failed: {type(e).__name__}: {e}")
            continue
        if rec is None or "error" in rec:
            print(f"[{sym}] {rec.get('error') if rec else 'no result'}")
            continue

        score, chain, short_q, long_q = rec["score"], rec["chain"], rec["short_q"], rec["long_q"]
        if score.vetoed:
            print(f"[{sym}] vetoed: {'; '.join(score.vetoes)}")
            continue
        if score.composite < threshold or score.coverage < min_coverage:
            print(f"[{sym}] below threshold: {score.composite:+.3f} "
                  f"(coverage {score.coverage:.0%})")
            continue

        token = nt.new_token()
        while token in pending:
            token = nt.new_token()
        try:
            trade = ex.size_trade(token, sym, chain, short_q, long_q, limits,
                                  open_risk=executor.open_risk())
        except ex.RiskRefusal as e:
            print(f"[{sym}] not sized: {e}")
            ex.record_refusal(token, str(e))
            continue

        notifier.send(nt.format_proposal(trade, score, limits), token=token)
        rec_out = {k: (v.isoformat() if isinstance(v, date) else v)
                   for k, v in asdict(trade).items()}
        rec_out["composite"] = score.composite
        pending[token] = rec_out
        print(f"[{sym}] proposed {token}: {trade.summary()}")


def _evaluate(symbol, clients, cal):
    """Score a symbol and return the concrete legs, or an error dict."""
    stats = ev.underlying_stats(clients["stock"], symbol)
    if stats is None:
        return {"error": "no underlying history"}
    chain = ev.live_chain(clients["option"], symbol, stats.spot, max_dte=60)
    if chain is None:
        return {"error": "no live two-sided market"}

    iv_atm = ev.atm_iv(chain, 30)
    slope = ev.term_structure_slope(chain)
    burst = ev.news_burst(clients["news"], symbol)
    exdiv, exdiv_src = ev.next_ex_dividend(clients["ca"], symbol)
    expiry = chain.nearest_expiry(45)
    if expiry is None:
        return {"error": "no expiry near 45 dte"}

    vetoes = []
    nxt = None if symbol in ev.NO_EARNINGS_SYMBOLS else cal.next_earnings(symbol)
    if nxt == ev.UNKNOWN:
        vetoes.append("earnings UNKNOWN")
    elif isinstance(nxt, date) and nxt <= expiry:
        vetoes.append(f"earnings {nxt} before expiry")
    # Only short CALLS carry dividend early-assignment risk; see scanner.py.
    _has_short_call = any(l.action == "sell" and l.right == "call" for l in pos.legs)
    if _has_short_call and exdiv and exdiv <= expiry:
        vetoes.append(f"ex-div {exdiv} ({exdiv_src}) before expiry")

    # Must match the deployed SSM config or the local path trades a
    # different strategy than the Lambdas, on the same account.
    strat = PutCreditSpread(short_delta=0.30, wing_width=3, target_dte=3)
    pos = strat.propose_entry(chain, 0)
    if pos is None:
        return {"error": "no strike pair matched"}
    short_leg = max(pos.legs, key=lambda l: l.strike)
    long_leg = min(pos.legs, key=lambda l: l.strike)
    sq = chain.find("put", short_leg.strike, expiry)
    lq = chain.find("put", long_leg.strike, expiry)
    if sq is None or lq is None:
        return {"error": "legs vanished from chain"}

    credit = (sq.bid - lq.ask) * 100
    half = (0.5 * (sq.ask - sq.bid) + 0.5 * (lq.ask - lq.bid)) * 100
    if credit > 0:
        sv = W.spread_veto(half, credit)
        if sv:
            vetoes.append(sv)
    else:
        vetoes.append("no net credit")

    from scanner import iv_percentile, record_iv
    series = record_iv(symbol, iv_atm) if iv_atm else []
    pctile = iv_percentile(series, iv_atm) if iv_atm else None
    score = W.score({
        "iv_minus_rv":    W.norm_iv_minus_rv(iv_atm, stats.realized_vol_20d),
        "iv_rank":        W.norm_iv_rank(pctile, len(series)),
        "term_slope":     W.norm_term_slope(slope),
        "trend_stress":   W.norm_trend_stress(stats.drawdown_20d, stats.pct_from_20d_ma),
        "spread_quality": W.norm_spread_quality(half, credit if credit > 0 else None),
        "news_intensity": W.norm_news_intensity(burst.intensity if burst else None),
    }, vetoes)
    return {"score": score, "chain": chain, "short_q": sq, "long_q": lq}


# --------------------------------------------------------------------------- #
def process_replies(notifier, executor, limits, pending, since, dry_run) -> None:
    if not pending:
        return
    try:
        replies = notifier.poll_replies(list(pending), since)
    except Exception as e:
        print(f"reply poll failed: {type(e).__name__}: {e}")
        return

    for r in replies:
        rec = pending.get(r.token)
        if rec is None:
            continue
        if not r.confirmed:
            print(f"[{r.token}] declined")
            ex.record_refusal(r.token, f"declined by SMS: {r.raw!r}")
            del pending[r.token]
            notifier.send(f"[{r.token}] skipped. Nothing submitted.")
            continue

        trade = _to_trade({k: v for k, v in rec.items() if k != "composite"})
        if r.contracts is not None and r.contracts != trade.contracts:
            # An altered size is a REQUEST, re-validated against the caps.
            requested = r.contracts
            capped = min(requested, limits.max_contracts)
            new_risk = capped * trade.max_loss_per_contract
            if new_risk > limits.max_open_risk:
                capped = int(limits.max_open_risk // trade.max_loss_per_contract)
            if capped < 1:
                notifier.send(f"[{r.token}] {requested} contracts exceeds limits. Not sent.")
                ex.record_refusal(r.token, f"requested {requested}, limits allow none")
                del pending[r.token]
                continue
            if capped != requested:
                notifier.send(f"[{r.token}] {requested} exceeds caps, using {capped}.")
            trade.contracts = capped
            trade.total_credit = capped * trade.credit_per_contract
            trade.total_risk = capped * trade.max_loss_per_contract

        try:
            out = executor.submit(trade, dry_run=dry_run)
            if out["status"] == "dry_run":
                notifier.send(f"[{r.token}] DRY RUN — would submit {trade.contracts} "
                              f"@ ${out['limit_price']:.2f} credit. No order placed.")
                print(f"[{r.token}] dry run: {out['limit_price']}")
            else:
                notifier.send(f"[{r.token}] SUBMITTED {trade.contracts}x "
                              f"@ ${out['limit_price']:.2f} credit "
                              f"({out['mode']}). Order {out['order_id'][:8]}")
                print(f"[{r.token}] submitted {out['order_id']}")
        except ex.RiskRefusal as e:
            notifier.send(f"[{r.token}] NOT submitted: {e}")
            ex.record_refusal(r.token, str(e))
            print(f"[{r.token}] refused: {e}")
        except Exception as e:
            notifier.send(f"[{r.token}] submit ERROR: {type(e).__name__}")
            ex.record_refusal(r.token, f"{type(e).__name__}: {e}")
            print(f"[{r.token}] error: {type(e).__name__}: {e}")
        del pending[r.token]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--sms", action="store_true", help="send real texts via Twilio")
    ap.add_argument("--ntfy", action="store_true", help="push via ntfy.sh instead of SMS")
    ap.add_argument("--arm", action="store_true", help="allow order submission")
    ap.add_argument("--live-money", action="store_true",
                    help="submit to the LIVE account instead of paper")
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything but never submit")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--min-coverage", type=float, default=0.60)
    ap.add_argument("--earnings-csv", default="earnings.csv")
    ap.add_argument("--equity", type=float, default=25_000.0)
    ap.add_argument("--risk-frac", type=float, default=0.02)
    ap.add_argument("--max-contracts", type=int, default=2)
    args = ap.parse_args()

    limits = ex.RiskLimits(account_equity=args.equity,
                           risk_per_trade_frac=args.risk_frac,
                           max_contracts=args.max_contracts)

    key = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        sys.exit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")

    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    paper = not args.live_money
    trading = TradingClient(key, sec, paper=paper)
    clients = {
        "option": OptionHistoricalDataClient(key, sec),
        "stock": StockHistoricalDataClient(key, sec),
        "news": NewsClient(key, sec),
        "ca": CorporateActionsClient(key, sec),
        "trading": trading,
    }

    # Confirm the account really is what the flags claim before anything arms.
    try:
        acct = trading.get_account()
        acct_no = acct.account_number
    except Exception as e:
        sys.exit(f"Cannot reach the {'LIVE' if args.live_money else 'paper'} account: {e}")
    looks_paper = str(acct_no).startswith("PA")
    if args.live_money and looks_paper:
        sys.exit(f"--live-money given but {acct_no} is a PAPER account. These are "
                 "separate credentials; generate live keys from the live dashboard.")
    if not args.live_money and not looks_paper:
        sys.exit(f"Account {acct_no} does not look like paper but --live-money was "
                 "not given. Refusing to guess which account you meant.")

    if args.ntfy:
        notifier = nt.NtfyNotifier()
        if args.arm and args.live_money:
            print("WARNING: ntfy public topics have no sender authentication. Anyone who\n"
                  "         learns the reply topic can confirm a live-money order. Use\n"
                  "         Twilio, a self-hosted ntfy with auth, or keep this to paper.")
    elif args.sms:
        notifier = nt.TwilioNotifier()
    else:
        notifier = nt.ConsoleNotifier()
    executor = ex.Executor(trading, clients["option"], limits,
                           armed=args.arm and not args.dry_run,
                           live=args.live_money)
    cal = ev.EarningsCalendar(args.earnings_csv)

    mode = ("DRY RUN" if args.dry_run else
            ("ARMED -> " + ("LIVE MONEY" if args.live_money else "paper")) if args.arm
            else "alerts only (not armed)")
    print(f"alerts: account {acct_no} | {mode} | risk/trade "
          f"${limits.risk_budget():.0f} | max {limits.max_contracts} contracts")
    if ex.KILL_SWITCH.exists():
        print(f"NOTE: kill switch present at {ex.KILL_SWITCH.resolve()} — "
              "no orders will submit until removed.")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    since = datetime.now(timezone.utc) - timedelta(minutes=15)

    while True:
        pending = _load_pending()
        for tok in _expire_stale(pending, limits):
            print(f"[{tok}] expired unanswered")
        process_replies(notifier, executor, limits, pending, since, args.dry_run)
        try:
            if trading.get_clock().is_open:
                scan_and_propose(symbols, clients, cal, limits, notifier, executor,
                                 args.threshold, args.min_coverage, pending)
            else:
                print("market closed")
        except Exception as e:
            print(f"scan error: {type(e).__name__}: {e}")
        _save_pending(pending)
        since = datetime.now(timezone.utc) - timedelta(minutes=2)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
