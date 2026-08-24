"""Dealer net gamma exposure (NGE), after Baltussen, Da, Lammers & Martens
(JFE 142, 2021), equations 13-15.

WHY THIS EXISTS

The strongest result in the intraday-momentum literature is not a Sharpe ratio,
it is a switch. Splitting the S&P 500 on the sign of dealer net gamma:

    NGE <  0    beta = 6.63   t = 4.78   R^2 = 3.58%
    NGE >= 0    beta = 0.82   t = 1.03   R^2 = 0.05%

When dealers are net short gamma they must hedge INTO the move and their
trading is forced, which is the whole mechanism. When they are net long there
is no effect at all -- not a weaker one, none. A strategy that trades every day
is therefore taking the signal on roughly half the days where it does not exist.

CONSTRUCTION

    NGE = [ sum_calls (gamma * OI * 100 * S) - sum_puts (gamma * OI * 100 * S) ]
          / underlying market value

The sign convention encodes the standing assumption of this literature: dealers
are long every call and short every put. That is a real assumption and a
contested one -- it is the same prior behind the commercial GEX series, and it
misses OTC put buying entirely, which biases the measure POSITIVE. Baltussen
concede this: their NGE is positive on 3,158 days and negative on 2,930, and
they expect the true split to be more negative than that.

THE LIMITATION THAT SHAPES EVERYTHING HERE

Alpaca serves CURRENT open interest, not a history. So this cannot be
backtested today -- it can only be accumulated. Every call appends a row, and
the conditional test becomes possible once enough days exist. That is slow and
it is the honest position: the alternative is inventing an OI history, which
would produce a confident number about nothing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

CONTRACT_MULT = 100


@dataclass
class GammaSnapshot:
    as_of: str
    underlying: str
    spot: float
    call_gamma_dollars: float
    put_gamma_dollars: float
    net_gamma_dollars: float
    nge: float                      # scaled by underlying market value
    contracts_used: int
    contracts_missing_oi: int
    oi_as_of: Optional[str]
    market_value: float

    @property
    def dealers_short_gamma(self) -> bool:
        return self.net_gamma_dollars < 0


def _all_contracts(trading, underlying: str, days_out: int = 60) -> List:
    """Page through every listed contract inside the window. The endpoint caps
    at 10k per page and the SPY chain alone runs to five figures, so paging is
    mandatory rather than defensive."""
    from alpaca.trading.requests import GetOptionContractsRequest
    out, token = [], None
    while True:
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date_gte=date.today(),
            expiration_date_lte=date.today() + timedelta(days=days_out),
            limit=10_000, page_token=token)
        r = trading.get_option_contracts(req)
        out.extend(r.option_contracts)
        token = getattr(r, "next_page_token", None)
        if not token:
            return out


def compute(trading, option_client, stock_client, underlying: str = "SPY",
            days_out: int = 60, shares_outstanding: float = None) -> GammaSnapshot:
    """Join current OI (trading API) against current gamma (market data API)."""
    from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest

    spot = float(stock_client.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=underlying))[underlying].price)

    oi: Dict[str, int] = {}
    oi_date = None
    for c in _all_contracts(trading, underlying, days_out):
        v = getattr(c, "open_interest", None)
        if v not in (None, ""):
            oi[c.symbol] = int(float(v))
            oi_date = oi_date or getattr(c, "open_interest_date", None)

    chain = option_client.get_option_chain(OptionChainRequest(underlying_symbol=underlying))

    call_g = put_g = 0.0
    used = missing = 0
    for sym, snap in chain.items():
        g = getattr(snap, "greeks", None)
        gamma = getattr(g, "gamma", None) if g else None
        if gamma is None:
            continue
        n = oi.get(sym)
        if n is None:
            missing += 1
            continue
        # dollar gamma: change in dealer delta per 1% underlying move
        dollars = float(gamma) * n * CONTRACT_MULT * spot
        # right is the char at -9 of the OCC symbol
        if sym[-9].upper() == "C":
            call_g += dollars
        else:
            put_g += dollars
        used += 1

    net = call_g - put_g
    mv = (shares_outstanding or 1e9) * spot
    return GammaSnapshot(
        as_of=datetime.now(timezone.utc).isoformat(),
        underlying=underlying, spot=spot,
        call_gamma_dollars=call_g, put_gamma_dollars=put_g,
        net_gamma_dollars=net, nge=net / mv if mv else 0.0,
        contracts_used=used, contracts_missing_oi=missing,
        oi_as_of=str(oi_date) if oi_date else None, market_value=mv)


def record(snap: GammaSnapshot) -> None:
    """Append to the history that makes the conditional test possible later."""
    import storage
    storage.JOURNAL.append({"kind": "gamma_snapshot", **asdict(snap)})


if __name__ == "__main__":
    import argparse, json
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient

    ap = argparse.ArgumentParser(description="Compute dealer net gamma exposure")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--days-out", type=int, default=60)
    a = ap.parse_args()

    k, s = os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    snap = compute(TradingClient(k, s, paper=True),
                   OptionHistoricalDataClient(k, s),
                   StockHistoricalDataClient(k, s),
                   a.symbol, a.days_out)
    print(json.dumps(asdict(snap), indent=2))
    print(f"\ndealers are {'SHORT' if snap.dealers_short_gamma else 'LONG'} gamma"
          f"  -> intraday momentum {'ON' if snap.dealers_short_gamma else 'OFF'}")
