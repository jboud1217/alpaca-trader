"""Strategy templates.

A Strategy is just two methods:

    propose_entry(chain, open_count) -> Position | None
    manage(position, chain)          -> close_reason (str) | None

The engine handles fills, MTM, and expiry. Strategies only decide *what* to
open and *when* to close. Everything a real research process wants to sweep --
target delta, wing width, DTE, profit-take %, stop multiple, min-DTE exit -- is
a constructor parameter, so you can grid-search them from run_backtest.py.

These are HYPOTHESES to test or kill, not recommendations to trade. A good
result on synthetic data only tells you the machinery is sound.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from data import OptionChain
from trade import Leg, Position


class Strategy:
    name = "base"

    def propose_entry(self, chain: OptionChain, open_count: int) -> Optional[Position]:
        raise NotImplementedError

    def manage(self, position: Position, chain: OptionChain) -> Optional[str]:
        raise NotImplementedError

    # shared management logic: profit target / stop / min-DTE roll-off
    def _default_manage(self, position, chain, profit_take, stop_mult, min_dte,
                        expire_below_delta=None, abandon_delta=0.15):
        """`expire_below_delta` enables the expire-worthless exit.

        Buying back a spread costs the full closing half-spread on every leg.
        On the SPY run 32 of 44 exits paid that, and friction totalled ~13% of
        credit against ~15% expectancy -- so the exit side alone is a large
        share of the edge. A contract allowed to expire settles at intrinsic and
        concedes nothing.

        The obvious danger is that holding past min_dte means holding through
        the highest-gamma stretch of the position's life, which is exactly the
        risk min_dte existed to sidestep. So this only engages when the short
        leg is genuinely far out of the money (|delta| below the threshold), and
        `abandon_delta` re-arms the normal close the moment the underlying comes
        back at it. Cheap-to-close is not the test; safe-to-abandon is.
        """
        dte = (position.expiry - chain.as_of).days

        def q_for(leg):
            return chain.find(leg.right, leg.strike, leg.expiry)

        def mid_or_intrinsic(leg):
            q = q_for(leg)
            if q is not None:
                return q.mid
            from bsm import intrinsic
            return intrinsic(leg.right, chain.spot, leg.strike)

        short_legs = [l for l in position.legs if l.action == "sell"]
        deltas = [abs(q_for(l).delta) for l in short_legs if q_for(l) is not None]
        worst_short_delta = max(deltas) if deltas else None

        if dte <= min_dte:
            if expire_below_delta is not None:
                if any(d >= abandon_delta for d in deltas):
                    return "abandon: short delta breached"
                # `all` on an empty list is True by design: a short leg with no
                # quote at all has fallen below the chain's min_price, which
                # means far OTM and near worthless -- safe to let expire.
                if all(d < expire_below_delta for d in deltas):
                    return None
            return f"min_dte<= {min_dte}"

        pnl = position.open_pnl(mid_or_intrinsic)
        credit = position.credit_received
        if credit > 0 and profit_take is not None and pnl >= profit_take * credit:
            return f"profit_target {int(profit_take * 100)}%"
        if credit > 0 and pnl <= -stop_mult * credit:
            return f"stop {stop_mult}x"
        return None


class PutCreditSpread(Strategy):
    """Sell an OTM put, buy a further-OTM put. Bullish/neutral premium selling."""

    name = "put_credit_spread"

    def __init__(
        self,
        short_delta: float = 0.30,
        wing_width: float = 5.0,
        target_dte: int = 45,
        profit_take: float = 0.50,
        stop_mult: float = 2.0,
        min_dte: int = 21,
        qty: int = 1,
        expire_below_delta: float = None,   # ride to expiry below this short delta
        abandon_delta: float = 0.15,        # ...unless it comes back at us
    ):
        self.short_delta = short_delta
        self.wing_width = wing_width
        self.target_dte = target_dte
        self.profit_take = profit_take
        self.stop_mult = stop_mult
        self.min_dte = min_dte
        self.qty = qty
        self.expire_below_delta = expire_below_delta
        self.abandon_delta = abandon_delta

    def propose_entry(self, chain, open_count):
        expiry = chain.nearest_expiry(self.target_dte)
        if expiry is None:
            return None
        short = chain.select_by_delta(expiry, "put", self.short_delta)
        if short is None:
            return None
        long = chain.leg_at_offset(expiry, "put", short.strike, self.wing_width)
        if long is None or long.strike >= short.strike:
            return None
        legs = [
            Leg("put", short.strike, expiry, "sell", self.qty, short.bid),   # sold: fill lower
            Leg("put", long.strike, expiry, "buy", self.qty, long.ask),      # bought: fill higher
        ]
        # NOTE: entry_price here is the raw quote side; the engine re-fills with
        # the slippage model so accounting stays in one place. We pass the
        # conservative side just so a Position is well-formed if inspected early.
        # Record the entry conditions, not just the delta. Without short_iv and
        # the spot/strike context there is no way to ask afterwards whether
        # entries at high implied vol did better than entries at low -- and
        # that conditioning question cannot be answered retroactively, because
        # the chain is gone once the backtest moves on.
        return Position(legs, chain.underlying, chain.as_of, tag=self.name,
                        meta={"short_delta": short.delta,
                              "short_iv": short.iv,
                              "short_strike": short.strike,
                              "long_iv": long.iv,
                              "spot": chain.spot,
                              "dte": (expiry - chain.as_of).days,
                              "credit": short.bid - long.ask})

    def manage(self, position, chain):
        return self._default_manage(position, chain, self.profit_take,
                                    self.stop_mult, self.min_dte,
                                    self.expire_below_delta, self.abandon_delta)


class IronCondor(Strategy):
    """Short put spread + short call spread. Defined risk on both sides."""

    name = "iron_condor"

    def __init__(
        self,
        short_delta: float = 0.20,
        wing_width: float = 5.0,
        target_dte: int = 45,
        profit_take: float = 0.50,
        stop_mult: float = 2.0,
        min_dte: int = 21,
        qty: int = 1,
        expire_below_delta: float = None,   # ride to expiry below this short delta
        abandon_delta: float = 0.15,        # ...unless it comes back at us
    ):
        self.short_delta = short_delta
        self.wing_width = wing_width
        self.target_dte = target_dte
        self.profit_take = profit_take
        self.stop_mult = stop_mult
        self.min_dte = min_dte
        self.qty = qty
        self.expire_below_delta = expire_below_delta
        self.abandon_delta = abandon_delta

    def propose_entry(self, chain, open_count):
        expiry = chain.nearest_expiry(self.target_dte)
        if expiry is None:
            return None
        sp = chain.select_by_delta(expiry, "put", self.short_delta)
        sc = chain.select_by_delta(expiry, "call", self.short_delta)
        if sp is None or sc is None:
            return None
        lp = chain.leg_at_offset(expiry, "put", sp.strike, self.wing_width)
        lc = chain.leg_at_offset(expiry, "call", sc.strike, self.wing_width)
        if lp is None or lc is None:
            return None
        if lp.strike >= sp.strike or lc.strike <= sc.strike:
            return None
        legs = [
            Leg("put", sp.strike, expiry, "sell", self.qty, sp.bid),
            Leg("put", lp.strike, expiry, "buy", self.qty, lp.ask),
            Leg("call", sc.strike, expiry, "sell", self.qty, sc.bid),
            Leg("call", lc.strike, expiry, "buy", self.qty, lc.ask),
        ]
        return Position(legs, chain.underlying, chain.as_of, tag=self.name)

    def manage(self, position, chain):
        return self._default_manage(position, chain, self.profit_take,
                                    self.stop_mult, self.min_dte,
                                    self.expire_below_delta, self.abandon_delta)
