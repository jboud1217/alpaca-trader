"""Backtest engine: fill model + portfolio + the daily event loop.

The fill model is the single most important honesty knob in the whole system.
Backtests lie by assuming you trade at mid. You don't -- you give up part of the
bid/ask spread every time, and on illiquid options that spread can dwarf the
edge. `slippage_frac` controls how much of the half-spread you concede beyond
mid: 0.0 = optimistic mid fills, 1.0 = you always cross to the bid/ask. Tune it
PESSIMISTICALLY. If a strategy only works at slippage_frac=0, it doesn't work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List

from bsm import intrinsic
from data import OptionsDataSource, OptionChain
from trade import CONTRACT_MULT, ClosedTrade, Leg, Position


@dataclass
class FillModel:
    slippage_frac: float = 0.5      # fraction of half-spread conceded beyond mid
    per_contract_fee: float = 0.05  # Alpaca equity options are commission-free;
                                    # this covers regulatory/exchange fees per contract

    def fill_price(self, quote, action: str) -> float:
        mid = quote.mid
        half = 0.5 * (quote.ask - quote.bid)
        if action == "sell":
            return max(0.0, mid - self.slippage_frac * half)
        return mid + self.slippage_frac * half


@dataclass
class BacktestConfig:
    underlying: str
    start: date
    end: date
    starting_cash: float = 25_000.0
    max_concurrent: int = 1
    one_entry_per_day: bool = True


class Backtest:
    def __init__(self, data: OptionsDataSource, strategy, config: BacktestConfig,
                 fill_model: FillModel = None):
        self.data = data
        self.strategy = strategy
        self.cfg = config
        self.fills = fill_model or FillModel()

        self.cash = config.starting_cash
        self.realized_pnl = 0.0
        self.open_positions: List[Position] = []
        self.closed: List[ClosedTrade] = []
        self.equity_curve: List[tuple] = []

        # Friction accounting. Every dollar conceded to the bid/ask, split by
        # side, because the two are fixed by different decisions: entry cost is
        # set by the structure you choose, exit cost by whether you buy the
        # position back at all. On the SPY run friction ran ~13% of credit
        # against a measured expectancy of ~15%, so this is not a rounding
        # detail -- it is nearly the whole result.
        self.spread_cost_open = 0.0
        self.spread_cost_close = 0.0
        self.credit_collected = 0.0
        self.expired_worthless = 0

    @property
    def spread_cost(self) -> float:
        return self.spread_cost_open + self.spread_cost_close

    @property
    def friction_frac(self) -> float:
        """Spread paid as a fraction of gross credit collected."""
        return self.spread_cost / self.credit_collected if self.credit_collected else 0.0

    # ---- fills ------------------------------------------------------------
    def _fee(self, position: Position) -> float:
        contracts = sum(l.qty for l in position.legs)
        return contracts * self.fills.per_contract_fee

    def _open(self, proposal: Position, chain: OptionChain, day: date):
        filled_legs = []
        for leg in proposal.legs:
            q = chain.find(leg.right, leg.strike, leg.expiry)
            if q is None:
                return  # can't fill cleanly; skip the trade rather than fake it
            price = self.fills.fill_price(q, leg.action)
            self.spread_cost_open += abs(price - q.mid) * leg.qty * CONTRACT_MULT
            filled_legs.append(Leg(leg.right, leg.strike, leg.expiry,
                                   leg.action, leg.qty, price))
        pos = Position(filled_legs, proposal.underlying, day,
                       tag=proposal.tag, meta=dict(proposal.meta))
        self.cash += pos.credit_received - self._fee(pos)
        self.credit_collected += max(pos.credit_received, 0.0)
        self.open_positions.append(pos)

    def _close(self, pos: Position, chain: OptionChain, day: date, reason: str):
        expired = day >= pos.expiry

        def close_price(leg: Leg) -> float:
            if expired:
                # Settlement, not a trade: you concede no spread letting a
                # contract expire. This is why an expire-worthless exit is
                # strictly cheaper than buying the position back.
                return intrinsic(leg.right, chain.spot, leg.strike)
            q = chain.find(leg.right, leg.strike, leg.expiry)
            if q is None:
                return intrinsic(leg.right, chain.spot, leg.strike)
            reverse = "buy" if leg.action == "sell" else "sell"
            price = self.fills.fill_price(q, reverse)
            self.spread_cost_close += abs(price - q.mid) * leg.qty * CONTRACT_MULT
            return price

        cost = pos.cost_to_close(close_price)   # net debit to flatten

        # A defined-risk spread cannot cost more than its width to close, nor
        # less than zero -- arbitrage forbids both. The MODELLED bid/ask can
        # violate that, because it adds a half-spread to each leg independently
        # and deep-ITM legs are expensive: a 750/753 spread was priced at $368
        # to close against a $300 structural ceiling, producing a loss $60 worse
        # than the maximum the position could possibly incur. Twelve of 280
        # trades were affected. Clamp to the no-arbitrage band so the tail is
        # not overstated by an artefact of the spread model. Fees are applied
        # separately below and legitimately sit outside this bound.
        widths = []
        for right in ("put", "call"):
            ks = sorted(l.strike for l in pos.legs if l.right == right)
            if len(ks) >= 2:
                widths.append((max(ks) - min(ks)) * CONTRACT_MULT)
        if widths:
            qty = max(l.qty for l in pos.legs)
            cap = max(widths) * qty
            cost = min(max(cost, 0.0), cap)
        # Expiry costs no commission either -- only a real closing trade does.
        closing_fee = 0.0 if expired else self._fee(pos)
        if expired:
            self.expired_worthless += 1
        pnl = pos.credit_received - cost - self._fee(pos) - closing_fee
        self.cash += -cost - closing_fee
        self.realized_pnl += pnl
        self.closed.append(ClosedTrade(pos, day, pnl, reason))
        self.open_positions.remove(pos)

    # ---- mark to market ---------------------------------------------------
    def _equity(self, chain: OptionChain) -> float:
        def mid_or_intrinsic(leg):
            q = chain.find(leg.right, leg.strike, leg.expiry)
            return q.mid if q is not None else intrinsic(leg.right, chain.spot, leg.strike)

        unrealized = sum(p.open_pnl(mid_or_intrinsic) for p in self.open_positions)
        return self.cfg.starting_cash + self.realized_pnl + unrealized

    # ---- main loop --------------------------------------------------------
    def run(self):
        for day in self.data.trading_days(self.cfg.start, self.cfg.end):
            chain = self.data.get_chain(self.cfg.underlying, day)
            if chain is None:
                continue

            # 1) manage / close existing positions first
            for pos in list(self.open_positions):
                reason = self.strategy.manage(pos, chain)
                if reason is None and day >= pos.expiry:
                    reason = "expiry"
                if reason:
                    self._close(pos, chain, day, reason)

            # 2) look for a new entry
            if len(self.open_positions) < self.cfg.max_concurrent:
                proposal = self.strategy.propose_entry(chain, len(self.open_positions))
                if proposal is not None:
                    self._open(proposal, chain, day)

            # 3) record equity
            self.equity_curve.append((day, self._equity(chain)))

        return self
