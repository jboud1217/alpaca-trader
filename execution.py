"""Order staging, sizing, and submission. The only module that can move money.

Everything here is written on the assumption that it will eventually be wrong
about something, so each control is independent and each one alone is enough to
stop a bad order:

    kill switch      a file on disk halts all submission, no restart needed
    armed flag       submission is off by default and must be turned on
    live confirm     pointing at live requires an explicit, separate opt-in
    per-trade cap    max contracts and max dollars of risk on one trade
    portfolio cap    max total open risk across everything
    daily caps       max orders/day and max new risk/day
    re-price gate    the quote is re-read at submit; a moved market aborts
    idempotency      client_order_id derives from the approval token
    limit only       never a market order on a multi-leg options spread

The re-price gate deserves the most attention because it is the one that is
easy to leave out and expensive to miss. A text goes out at 10:00 quoting $95 of
credit. You reply at 10:40. If the order submits at the old price it may now be
badly mispriced, and on a spread the mid can move faster than the underlying.
So the market is re-read at submission and the order is abandoned if the
available credit has slipped past a tolerance. An approval is consent to a
price, not a standing instruction.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import storage

# Kept as module attributes for backwards compatibility with local tooling and
# docs; the authoritative source is storage.KILL / storage.JOURNAL, which Lambda
# rebinds to DynamoDB at cold start.
KILL_SWITCH = Path("KILL_SWITCH")
ORDER_JOURNAL = Path("orders.jsonl")
CONTRACT_MULT = 100


def flag_on(cfg, name: str) -> bool:
    """Parse an SSM-style on/off flag. Canonical home is here, NOT in the AWS
    handler, so the safety interlocks that depend on it can be tested without
    importing boto3 or anything else AWS-shaped."""
    return str(cfg.get(name, "")).strip().lower() in ("on", "true", "1", "yes")


def opt_int(value, default=None) -> Optional[int]:
    """Parse an OPTIONAL integer cap. "", "none", "off", "unlimited" and any
    non-positive number all mean NO CAP (None).

    Canonical home is here beside flag_on, and for the same reason: a cap that
    can silently parse to the wrong thing is a safety control, so it must be
    testable without importing boto3.
    """
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("", "none", "off", "unlimited", "no", "null"):
        return None
    try:
        n = int(float(s))
    except (TypeError, ValueError):
        return default
    return n if n > 0 else None


def auto_accept_effective(cfg) -> bool:
    """Whether proposals should be auto-confirmed.

    THE INTERLOCK: auto_accept is a paper-only convenience and is ignored
    whenever live_money is on. The hazard was never auto-accept itself -- it is
    auto-accept SURVIVING a flip to live, leaving an unattended trader spending
    real money. Encoding that here means it cannot be forgotten, and means it
    can be tested.
    """
    return flag_on(cfg, "auto_accept") and not flag_on(cfg, "live_money")


class RiskRefusal(Exception):
    """Raised when a control blocks an order. Never caught silently."""


# --------------------------------------------------------------------------- #
@dataclass
class RiskLimits:
    """Hard caps. These are refusals, not preferences -- nothing overrides them
    except editing this object, which is a deliberate act."""

    account_equity: float = 25_000.0
    risk_per_trade_frac: float = 0.02      # 2% of equity at risk on one trade
    max_contracts: int = 2                 # absolute ceiling per order
    max_risk_per_trade: float = 750.0      # dollars, absolute ceiling
    max_open_risk: float = 2_500.0         # total defined risk across open positions
    # Per-underlying ceiling, as a fraction of max_open_risk. Without this the
    # portfolio cap gives an ILLUSION of diversification: it was satisfied while
    # the book held two IWM spreads at 302/299 and 303/300 -- adjacent strikes,
    # same name, same expiry. Those are not two positions. One gap below 299
    # takes both to max loss simultaneously.
    #
    # It matters more here than it looks, because the three symbols traded are
    # SPY/QQQ/IWM at pairwise correlation ~0.87 -- about 1.1 independent assets
    # even when fully spread out. Concentrating inside one name throws away the
    # little diversification that exists.
    max_open_risk_per_underlying_frac: float = 0.50
    # None means the NUMBER of orders is not capped. The count cap was removed
    # as a control because it measured the wrong thing: closes consume it too,
    # so a day of exits could lock out entries while almost no capital was at
    # risk. On 2026-09-03 a self-defeating open/close loop spent six of eight
    # slots in five minutes and refused every later proposal, having deployed
    # and returned the money within the same five minutes. What matters is
    # capital at risk, which max_new_risk_per_day enforces directly.
    max_orders_per_day: Optional[int] = None
    # Capital the day may DEPLOY, net of what came back. Consumed budget is
    # (defined risk opened today) - (net realised P&L today), so a winner frees
    # room for the next trade and a loser tightens it. Note this is a NET
    # figure by deliberate choice: a day that loses money gets less rope, not
    # the same rope.
    max_new_risk_per_day: float = 1_000.0
    min_credit_per_contract: float = 20.0  # below this, fees and spread dominate
    max_spread_frac_of_credit: float = 0.15
    reprice_tolerance_frac: float = 0.10   # abort if credit slipped >10% since the text
    approval_ttl_seconds: int = 1800       # a 30-min-old approval is stale, not consent
    # How hard to cross the spread on entry. 1.0 prices at short.bid - long.ask,
    # fully marketable, near-certain fill, and you concede the whole spread --
    # which the friction sweep measured at 7.3% of credit. 0.5 prices at the
    # mid and halves that, at the cost of possibly not filling. Same knob as
    # FillModel.slippage_frac in the backtest, so what you sweep there is what
    # you get here. Defaults to certainty because a partially-filled multi-leg
    # spread is a worse problem than a slightly worse price.
    limit_slippage_frac: float = 1.0
    # The exit rules, mirrored here so the sizing path can check whether they
    # are arithmetically survivable. They live in the strategy too; this copy
    # exists because size_trade is where a trade can still be refused.
    profit_take: Optional[float] = 0.50
    stop_mult: Optional[float] = 2.0
    # How much worse than the delta-implied rate the break-even may be before
    # refusing. 0.0 would refuse almost everything (delta is not a precise
    # probability); this allows a modest gap and blocks the egregious case.
    breakeven_slack: float = 0.10

    def risk_budget(self) -> float:
        return min(self.account_equity * self.risk_per_trade_frac,
                   self.max_risk_per_trade)


@dataclass
class SizedTrade:
    """A concrete, priced, sized proposal. Immutable once texted."""
    token: str
    symbol: str
    underlying: str
    expiry: date
    short_strike: float
    long_strike: float
    short_occ: str
    long_occ: str
    contracts: int
    credit_per_contract: float             # dollars
    max_loss_per_contract: float           # dollars
    total_credit: float
    total_risk: float
    short_delta: float
    half_spread_cost: float
    created_at: str
    sizing_note: str = ""

    def summary(self) -> str:
        return (f"{self.underlying} {self.short_strike:.0f}/{self.long_strike:.0f}p "
                f"{self.expiry:%b%d} x{self.contracts}  "
                f"credit ${self.total_credit:.0f}  risk ${self.total_risk:.0f}")


# --------------------------------------------------------------------------- #
def _split_occ(symbol: str):
    """OCC symbol -> (root, expiry, right, strike), or Nones if unparseable.

    Parsed from the RIGHT because the root is variable length:
    <ROOT><YYMMDD><C|P><strike * 1000, zero-padded to 8>.
    IWM260817P00302000 -> ("IWM", "260817", "P", 302.0)
    """
    try:
        strike = int(symbol[-8:]) / 1000.0
        right = symbol[-9].upper()
        if right not in ("C", "P"):
            return None, None, None, None
        expiry = symbol[-15:-9]
        root = symbol[:-15].upper()
        if not root or not root.isalpha() or not expiry.isdigit():
            return None, None, None, None
        return root, expiry, right, strike
    except (ValueError, IndexError):
        return None, None, None, None


def occ_symbol(underlying: str, expiry: date, right: str, strike: float) -> str:
    """<ROOT><YYMMDD><C|P><strike*1000 zero-padded to 8>."""
    return (f"{underlying.upper()}{expiry:%y%m%d}"
            f"{'C' if right.lower() == 'call' else 'P'}"
            f"{int(round(strike * 1000)):08d}")


def breakeven_win_rate(credit: float, max_loss: float,
                       profit_take=None, stop_mult=None) -> float:
    """The win rate an exit rule set REQUIRES in order to break even.

    This is the number that explained the live results. With
    profit_take=0.50 and stop_mult=2.0 on a ~$50 credit:

        win  is capped at  0.50 x 50  =  $25
        loss is capped at  2.00 x 50  = $100

    a 4:1 asymmetry that needs ~80% wins. The measured rate was 50%, because
    at 3 DTE the gamma is large enough to trip a 2x stop long before expiry
    decides anything. The entry signal was never the binding constraint -- the
    exits were, and nothing in the sizing path was checking that.
    """
    win = profit_take * credit if profit_take else credit
    loss = min(stop_mult * credit, max_loss) if stop_mult else max_loss
    if win <= 0 or loss <= 0:
        return 1.0
    return loss / (loss + win)


def implied_win_rate(short_delta: float) -> float:
    """P(short leg finishes OTM), approximated by 1 - |delta|. Rough -- delta
    is a risk-neutral probability and carries a drift term -- but the error is
    small next to the gap this exists to catch."""
    return max(0.0, min(1.0, 1.0 - abs(short_delta)))


def size_trade(token: str, underlying: str, chain, short_q, long_q,
               limits: RiskLimits, open_risk: float = 0.0,
               open_risk_this_underlying: float = 0.0) -> SizedTrade:
    """Turn a chain proposal into a contract count, or refuse.

    Sizes off DEFINED RISK, not notional and not buying power. For a credit
    spread the true exposure is (width - credit) per contract, which is the
    number that actually shows up if it goes wrong, so that is the number the
    budget is divided by.
    """
    if short_q.symbol is None or long_q.symbol is None:
        raise RiskRefusal(
            "quotes carry no OCC symbol -- this chain did not come from live "
            "market data, and orders are never built from historical or "
            "synthetic quotes")

    credit = (short_q.bid - long_q.ask) * CONTRACT_MULT
    if credit <= 0:
        raise RiskRefusal(f"no net credit at live quotes (${credit:.0f})")
    if credit < limits.min_credit_per_contract:
        raise RiskRefusal(f"credit ${credit:.0f}/contract below floor "
                          f"${limits.min_credit_per_contract:.0f}")

    width = abs(short_q.strike - long_q.strike) * CONTRACT_MULT
    max_loss = width - credit
    if max_loss <= 0:
        raise RiskRefusal(f"implied max loss ${max_loss:.0f} is not positive -- "
                          "quotes look wrong, refusing rather than guessing")

    half_spread = (0.5 * (short_q.ask - short_q.bid)
                   + 0.5 * (long_q.ask - long_q.bid)) * CONTRACT_MULT
    frac = half_spread / credit
    if frac > limits.max_spread_frac_of_credit:
        raise RiskRefusal(f"round-trip spread {frac:.0%} of credit exceeds "
                          f"{limits.max_spread_frac_of_credit:.0%}")

    # Do the exit-rule arithmetic BEFORE sizing. If the configured
    # profit_take/stop_mult demand a higher win rate than the chosen delta can
    # plausibly deliver, the trade is negative-expectancy by construction and no
    # entry signal rescues it. Measured live at 50% wins against an 81%
    # requirement; this refuses that shape rather than trading it.
    # INFORMATIONAL, not a refusal -- and the reason matters.
    #
    # The nominal caps overstate the requirement, because most trades never
    # reach either one: they exit at expiry or min_dte first. Nominal says the
    # deployed rules need 80% wins; the backtest realises 71% wins at a 0.42
    # payoff ratio, which is break-even. Gating on the nominal number would
    # refuse every configuration tested, including the best one.
    #
    # It is recorded rather than dropped because the ratio it exposes is the
    # whole finding: a 30-delta short implies a 70% win rate, so a FAIR payoff
    # ratio is 0.30/0.70 = 0.43. The backtest measures 0.42 across all twelve
    # exit-rule variants. The market is pricing this at fair value -- there is
    # no premium being handed over, and no exit rule creates one.
    be = breakeven_win_rate(credit, max_loss,
                            limits.profit_take, limits.stop_mult)
    iw = implied_win_rate(short_q.delta)
    be_note = f"; break-even {be:.0%} vs {iw:.0%} delta-implied"

    budget = limits.risk_budget()
    contracts = int(math.floor(budget / max_loss))
    note = f"budget ${budget:.0f} / ${max_loss:.0f} risk per contract" + be_note
    if contracts > limits.max_contracts:
        contracts = limits.max_contracts
        note += f"; capped at max_contracts={limits.max_contracts}"
    if contracts < 1:
        raise RiskRefusal(
            f"risk budget ${budget:.0f} will not cover one contract at "
            f"${max_loss:.0f} of defined risk")

    total_risk = contracts * max_loss
    if open_risk + total_risk > limits.max_open_risk:
        room = limits.max_open_risk - open_risk
        contracts = int(math.floor(room / max_loss))
        if contracts < 1:
            raise RiskRefusal(
                f"portfolio risk cap: ${open_risk:.0f} already open against a "
                f"${limits.max_open_risk:.0f} ceiling, no room for this trade")
        total_risk = contracts * max_loss
        note += f"; reduced to fit ${limits.max_open_risk:.0f} portfolio cap"

    # Per-underlying cap, applied AFTER the portfolio cap so the tighter of the
    # two always wins.
    per_name_cap = limits.max_open_risk * limits.max_open_risk_per_underlying_frac
    if open_risk_this_underlying + total_risk > per_name_cap:
        room = per_name_cap - open_risk_this_underlying
        contracts = int(math.floor(room / max_loss))
        if contracts < 1:
            raise RiskRefusal(
                f"{underlying} concentration cap: ${open_risk_this_underlying:.0f} "
                f"already open in {underlying} against a ${per_name_cap:.0f} "
                f"per-underlying ceiling "
                f"({limits.max_open_risk_per_underlying_frac:.0%} of "
                f"${limits.max_open_risk:.0f}), no room for this trade")
        total_risk = contracts * max_loss
        note += f"; reduced to fit ${per_name_cap:.0f} {underlying} cap"

    return SizedTrade(
        token=token, symbol=underlying, underlying=underlying,
        expiry=short_q.expiry, short_strike=short_q.strike,
        long_strike=long_q.strike, short_occ=short_q.symbol,
        long_occ=long_q.symbol, contracts=contracts,
        credit_per_contract=credit, max_loss_per_contract=max_loss,
        total_credit=contracts * credit, total_risk=total_risk,
        short_delta=short_q.delta, half_spread_cost=half_spread,
        created_at=datetime.now(timezone.utc).isoformat(), sizing_note=note)


# --------------------------------------------------------------------------- #
class Executor:
    """Submits multi-leg option orders, subject to every control above."""

    def __init__(self, trading_client, option_client, limits: RiskLimits, *,
                 armed: bool = False, live: bool = False):
        self.trading = trading_client
        self.opt = option_client
        self.limits = limits
        self.armed = armed
        self.live = live

    # -- controls ---------------------------------------------------------- #
    def _check_kill_switch(self) -> None:
        if storage.KILL.engaged():
            where = getattr(storage.KILL, "describe", lambda: "kill switch")()
            raise RiskRefusal(f"kill switch engaged ({where}) -- clear it to resume")

    def _check_armed(self) -> None:
        if not self.armed:
            raise RiskRefusal("executor is not armed (pass --arm to enable submission)")

    def _todays_orders(self) -> List[dict]:
        return storage.JOURNAL.todays_submitted()

    def _check_daily_caps(self, trade: SizedTrade) -> None:
        today = self._todays_orders()

        # Optional, and off by default. Kept because a numeric value is still
        # a useful circuit breaker if an entry loop ever runs away again.
        cap = self.limits.max_orders_per_day
        if cap is not None and cap > 0 and len(today) >= cap:
            raise RiskRefusal(f"daily order cap reached ({len(today)}/{cap})")

        # Closes carry total_risk 0.0, so only entries count as deployment.
        deployed = sum(r.get("total_risk", 0.0) for r in today)
        try:
            realized = float(storage.JOURNAL.todays_realized_pnl())
        except (AttributeError, NotImplementedError):
            # An older journal implementation cannot answer this. Fall back to
            # gross deployment: stricter than intended, never looser.
            realized = 0.0
        consumed = deployed - realized
        if consumed + trade.total_risk > self.limits.max_new_risk_per_day:
            raise RiskRefusal(
                f"daily capital cap: ${deployed:.0f} deployed, "
                f"${realized:+.0f} realised, ${consumed:.0f} consumed; "
                f"this adds ${trade.total_risk:.0f}, ceiling "
                f"${self.limits.max_new_risk_per_day:.0f}")

    def _check_already_submitted(self, trade: SizedTrade) -> None:
        """Idempotency: one approval token can produce at most one order."""
        prior = storage.JOURNAL.find_submitted(trade.token)
        if prior is not None:
            raise RiskRefusal(f"token {trade.token} already submitted "
                              f"(order {prior.get('order_id')}) -- refusing duplicate")

    def _check_market_open(self) -> None:
        """Refuse outside RTH rather than let the broker reject it.

        A proposal texted at 15:50 and confirmed at 16:05 is inside its 30-minute
        TTL but outside the session. Submitting anyway produces an opaque API
        error and an alarming "submit ERROR" text for what is really a mundane
        situation. Options do not trade in extended hours, so this is a refusal
        with a readable reason.
        """
        try:
            if not self.trading.get_clock().is_open:
                raise RiskRefusal("market is closed -- options do not trade in "
                                  "extended hours; re-scan at the next open")
        except RiskRefusal:
            raise
        except Exception as e:
            raise RiskRefusal(f"cannot confirm market is open ({type(e).__name__}) "
                              "-- refusing rather than assuming")

    def _check_age(self, trade: SizedTrade) -> None:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(trade.created_at)).total_seconds()
        if age > self.limits.approval_ttl_seconds:
            raise RiskRefusal(
                f"proposal is {age/60:.0f} min old, past the "
                f"{self.limits.approval_ttl_seconds/60:.0f} min TTL -- "
                "re-scan rather than trade a stale quote")

    def _reprice(self, trade: SizedTrade) -> float:
        """Re-read the market at submission time. Consent was to a price."""
        from alpaca.data.requests import OptionLatestQuoteRequest
        q = self.opt.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=[trade.short_occ, trade.long_occ]))
        s, l = q.get(trade.short_occ), q.get(trade.long_occ)
        if s is None or l is None:
            raise RiskRefusal("no live quote for one or both legs at submit time")
        # Fully-crossing credit: sell the short at the bid, buy the long at the
        # ask. This is the worst price you would accept and the one that fills.
        credit_cross = (float(s.bid_price) - float(l.ask_price)) * CONTRACT_MULT
        credit_mid = (0.5 * (float(s.bid_price) + float(s.ask_price))
                      - 0.5 * (float(l.bid_price) + float(l.ask_price))) * CONTRACT_MULT
        f = self.limits.limit_slippage_frac
        credit_now = credit_mid - f * (credit_mid - credit_cross)
        if credit_now <= 0:
            raise RiskRefusal(f"credit has gone negative (${credit_now:.0f}) since the alert")
        slip = (trade.credit_per_contract - credit_now) / trade.credit_per_contract
        if slip > self.limits.reprice_tolerance_frac:
            raise RiskRefusal(
                f"credit moved from ${trade.credit_per_contract:.0f} to "
                f"${credit_now:.0f} ({slip:.0%} worse) since the alert, past the "
                f"{self.limits.reprice_tolerance_frac:.0%} tolerance")
        return credit_now

    def open_risk(self, underlying: str = None) -> float:
        """Total defined risk across open option positions.

        With `underlying` set, counts only that name -- which is what the
        per-underlying concentration cap needs. The OCC root is the leading
        alphabetic run of the contract symbol, so IWM260817P00302000 -> IWM.

        Approximates each short option's exposure by its strike notional when it
        cannot pair legs into spreads. That errs HIGH, which for a risk ceiling
        is the correct direction to be wrong in.
        """
        try:
            positions = self.trading.get_all_positions()
        except Exception:
            return float("inf")   # cannot verify exposure => behave as if full
        want = underlying.upper() if underlying else None

        # Pair legs into spreads before pricing the risk. The old fallback --
        # assume every naked short is 5-wide -- overstated a book of 3-wide
        # spreads by 67% ($1,500 against a true $900), which would park the
        # system at its ceiling with 40% of the budget actually free. Erring
        # high is the right direction for a ceiling, but not so high that the
        # ceiling stops being usable.
        #
        # Legs pair by (root, expiry, right). Each short is matched to the
        # nearest unused long on the same side; the width between them is the
        # real exposure. Anything left unmatched is genuinely naked and still
        # gets the conservative 5-wide assumption.
        shorts, longs = {}, {}
        for p in positions:
            ac = f"{getattr(p, 'asset_class', '')}" \
                 f"{getattr(getattr(p, 'asset_class', None), 'value', '')}".lower()
            if "option" not in ac:
                continue
            sym = str(getattr(p, "symbol", ""))
            root, expiry, right, strike = _split_occ(sym)
            if root is None:
                continue
            if want is not None and root != want:
                continue
            key = (root, expiry, right)
            bucket = shorts if float(p.qty) < 0 else longs
            for _ in range(int(abs(float(p.qty)))):
                bucket.setdefault(key, []).append(strike)

        risk = 0.0
        for key, short_strikes in shorts.items():
            available = sorted(longs.get(key, []))
            for k in sorted(short_strikes):
                if available:
                    # nearest long on the protective side
                    j = min(range(len(available)), key=lambda i: abs(available[i] - k))
                    risk += abs(k - available.pop(j)) * CONTRACT_MULT
                else:
                    risk += CONTRACT_MULT * 5.0   # naked: assume 5-wide, as before
        return risk

    def _legacy_open_risk(self, positions, want):
        risk = 0.0
        for p in positions:
            # CASE MATTERS HERE and it silently did not for a long time.
            # alpaca-py returns the AssetClass enum, whose str() is
            # "AssetClass.US_OPTION" -- uppercase. The original test was
            # `"option" in str(p.asset_class)`, which is never true, so this
            # method returned 0.0 with a book full of options and the
            # max_open_risk ceiling never once fired. Lowercase both sides, and
            # check .value too in case the enum's repr changes upstream.
            ac = f"{getattr(p, 'asset_class', '')}" \
                 f"{getattr(getattr(p, 'asset_class', None), 'value', '')}".lower()
            if "option" in ac:
                if want is not None:
                    sym = str(getattr(p, "symbol", ""))
                    root = "".join(c for c in sym[:6] if c.isalpha()).upper()
                    if root != want:
                        continue
                qty = abs(float(p.qty))
                if float(p.qty) < 0:
                    risk += qty * CONTRACT_MULT * 5.0    # assume 5-wide if unpaired
        return risk

    # -- submission -------------------------------------------------------- #
    def submit(self, trade: SizedTrade, *, dry_run: bool = False) -> dict:
        from alpaca.trading.enums import (OrderClass, OrderSide, PositionIntent,
                                          TimeInForce)
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

        # A dry run cannot place an order -- it returns below, before
        # submit_order is ever reached -- so the arming and kill-switch gates
        # are skipped for it. Otherwise `--dry-run` would refuse at the first
        # gate and never show you what it would have done, which is the entire
        # point of a dry run. Every gate that validates the TRADE still runs,
        # so a dry run exercises the same logic a real submission would.
        if not dry_run:
            self._check_kill_switch()
            self._check_armed()
            self._check_market_open()
        self._check_age(trade)
        self._check_already_submitted(trade)
        self._check_daily_caps(trade)

        open_risk = self.open_risk()
        if open_risk + trade.total_risk > self.limits.max_open_risk:
            raise RiskRefusal(
                f"portfolio cap: ${open_risk:.0f} open + ${trade.total_risk:.0f} new "
                f"> ${self.limits.max_open_risk:.0f}")

        credit_now = self._reprice(trade)

        # Limit at the re-priced credit, never a market order. A multi-leg
        # market order on options is an invitation to be filled at the far side
        # of every leg at once.
        limit_price = round(credit_now / CONTRACT_MULT, 2)

        req = LimitOrderRequest(
            qty=trade.contracts,
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=f"harness-{trade.token}",
            legs=[
                OptionLegRequest(symbol=trade.short_occ, ratio_qty=1,
                                 side=OrderSide.SELL,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
                OptionLegRequest(symbol=trade.long_occ, ratio_qty=1,
                                 side=OrderSide.BUY,
                                 position_intent=PositionIntent.BUY_TO_OPEN),
            ],
        )

        rec = {
            "token": trade.token,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "mode": "live" if self.live else "paper",
            "dry_run": dry_run,
            "trade": {k: (v.isoformat() if isinstance(v, date) else v)
                      for k, v in asdict(trade).items()},
            "limit_price": limit_price,
            "credit_at_alert": trade.credit_per_contract,
            "credit_at_submit": credit_now,
            "total_risk": trade.total_risk,
        }

        if dry_run:
            rec["status"] = "dry_run"
            _journal(rec)
            return rec

        order = self.trading.submit_order(req)
        rec["status"] = "submitted"
        rec["order_id"] = str(order.id)
        rec["order_status"] = str(getattr(order, "status", ""))
        _journal(rec)

        # Record the position so it can be MANAGED. The broker knows the legs
        # but not the credit collected, and both the profit target and the stop
        # are multiples of that credit -- without this an exit rule has nothing
        # to measure against.
        try:
            storage.POSITIONS.put(trade.token, {
                "token": trade.token, "underlying": trade.underlying,
                "expiry": trade.expiry.isoformat(),
                "short_occ": trade.short_occ, "long_occ": trade.long_occ,
                "short_strike": trade.short_strike, "long_strike": trade.long_strike,
                "contracts": trade.contracts,
                "entry_credit_per_contract": limit_price * CONTRACT_MULT,
                "max_loss_per_contract": trade.max_loss_per_contract,
                "opened_at": rec["submitted_at"], "order_id": rec["order_id"],
                "mode": rec["mode"],
            })
        except Exception as e:
            # Never fail the order because bookkeeping failed -- but say so
            # loudly, because an unrecorded position is an unmanaged one.
            print(f"WARNING: order {rec['order_id']} submitted but NOT recorded "
                  f"for management: {type(e).__name__}: {e}")
        return rec


def _journal(rec: dict) -> None:
    storage.JOURNAL.append(rec)


def record_refusal(token: str, reason: str) -> None:
    _journal({"token": token, "status": "refused", "reason": reason,
              "submitted_at": datetime.now(timezone.utc).isoformat()})
