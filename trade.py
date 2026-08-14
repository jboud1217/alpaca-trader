"""Core trade objects shared by strategies and the engine.

Sign convention (the thing most backtests get subtly wrong, so it's centralized
here): every leg contributes with a sign of +1 if we SOLD it and -1 if we
BOUGHT it. That single convention makes credit, cost-to-close, and P&L fall out
consistently for any combination of legs.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from bsm import intrinsic

CONTRACT_MULT = 100  # US equity/index options are 100 shares per contract


@dataclass
class Leg:
    right: str          # 'put' or 'call'
    strike: float
    expiry: date
    action: str         # 'sell' or 'buy' (the OPENING action)
    qty: int
    entry_price: float  # per-share premium at fill (already slippage-adjusted)

    @property
    def sign(self) -> int:
        return +1 if self.action == "sell" else -1


@dataclass
class Position:
    legs: List[Leg]
    underlying: str
    entry_date: date
    tag: str = ""                    # strategy name / variant label
    meta: dict = field(default_factory=dict)

    # ---- lifecycle-independent facts -------------------------------------
    @property
    def expiry(self) -> date:
        # Defined-risk templates here share one expiry across legs.
        return min(leg.expiry for leg in self.legs)

    @property
    def credit_received(self) -> float:
        """Net dollars collected at entry (positive for a credit structure)."""
        return sum(l.sign * l.entry_price * l.qty * CONTRACT_MULT for l in self.legs)

    @property
    def max_loss(self) -> float:
        """Defined-risk max loss in dollars for verticals / condors.

        Width of the widest spread minus credit. Assumes standard defined-risk
        construction; returns None if it can't infer a width.
        """
        widths = []
        for right in ("put", "call"):
            strikes = sorted(l.strike for l in self.legs if l.right == right)
            if len(strikes) >= 2:
                widths.append(max(strikes) - min(strikes))
        if not widths:
            return None
        worst_width = max(widths) * CONTRACT_MULT
        return worst_width - self.credit_received

    # ---- mark-to-market helpers ------------------------------------------
    def cost_to_close(self, price_fn) -> float:
        """Net debit (dollars) to flatten now. price_fn(leg) -> per-share price.

        To close we reverse each leg, so a sold leg costs us (we buy it back)
        and a bought leg pays us. Using the same +1/-1 sign, cost-to-close is
        just sign * price summed over legs.
        """
        return sum(l.sign * price_fn(l) * l.qty * CONTRACT_MULT for l in self.legs)

    def open_pnl(self, price_fn) -> float:
        """Unrealized P&L given a per-leg pricing function."""
        return self.credit_received - self.cost_to_close(price_fn)

    def intrinsic_price_fn(self, spot: float):
        return lambda l: intrinsic(l.right, spot, l.strike)


@dataclass
class ClosedTrade:
    position: Position
    exit_date: date
    pnl: float
    reason: str

    @property
    def days_held(self) -> int:
        return (self.exit_date - self.position.entry_date).days
