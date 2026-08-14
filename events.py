"""Real-time event sources for the live scanner.

WHAT ALPACA ACTUALLY GIVES YOU (verified against alpaca-py 0.43.x):

    news              /v1beta1/news              historical + live
    corporate actions /v1beta1/corporate-actions dividends, splits, mergers
    option snapshots  /v1beta1/options/snapshots REAL bid/ask + greeks + IV
    stock bars        /v2/stocks/bars            for realized vol and trend

    NOT AVAILABLE: earnings calendar, economic calendar (FOMC/CPI/NFP),
    analyst actions. There is no endpoint for any of them at any tier.

Earnings is the single most important scheduled event for single-name options,
so its absence is handled explicitly rather than ignored: `EarningsCalendar`
reads a CSV you maintain or wire to your own provider. If it has no data for a
symbol it reports UNKNOWN, and the scorer treats unknown-earnings as a reason to
withhold confidence rather than as "no earnings" -- silently assuming no
earnings is how you end up short gamma into a print.

ONE GENUINE ADVANTAGE OVER THE BACKTEST PATH: live option snapshots carry real
bid/ask and real greeks. The spread modelling that handicaps `--real` does not
apply here. Live suggestions are priced off actual quotes.
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from data import OptionChain, OptionQuote


# --------------------------------------------------------------------------- #
# Live option chain (real quotes, real greeks -- no spread model needed)
# --------------------------------------------------------------------------- #
def live_chain(option_client, underlying: str, spot: float, *,
               max_dte: int = 60, strike_band: float = 0.15,
               as_of: Optional[date] = None, feed=None) -> Optional[OptionChain]:
    """Build an OptionChain from live snapshots so the existing Strategy
    templates work unchanged against real-time data.

    Uses the snapshot's own delta and IV rather than reconstructing them, and
    its own bid/ask rather than modelling them. Contracts without a two-sided
    market are dropped: you cannot trade what nobody is quoting.
    """
    from alpaca.data.requests import OptionChainRequest

    as_of = as_of or date.today()
    kwargs = {
        "underlying_symbol": underlying,
        "expiration_date_gte": as_of + timedelta(days=1),
        "expiration_date_lte": as_of + timedelta(days=max_dte),
        "strike_price_gte": spot * (1 - strike_band),
        "strike_price_lte": spot * (1 + strike_band),
    }
    if feed:
        kwargs["feed"] = feed
    snaps = option_client.get_option_chain(OptionChainRequest(**kwargs))

    from spread import _parse_occ
    quotes: List[OptionQuote] = []
    for symbol, snap in snaps.items():
        q = getattr(snap, "latest_quote", None)
        g = getattr(snap, "greeks", None)
        iv = getattr(snap, "implied_volatility", None)
        if q is None or g is None or iv is None:
            continue
        if q.bid_price is None or q.ask_price is None:
            continue
        bid, ask = float(q.bid_price), float(q.ask_price)
        if bid <= 0 or ask <= bid:
            continue                        # no market, or crossed
        expiry, right, strike = _parse_occ(symbol)
        if expiry is None or g.delta is None:
            continue
        quotes.append(OptionQuote(right=right, strike=strike, expiry=expiry,
                                  bid=bid, ask=ask, delta=float(g.delta),
                                  iv=float(iv), symbol=symbol))
    if not quotes:
        return None
    return OptionChain(as_of, underlying, spot, quotes)


def atm_iv(chain: OptionChain, target_dte: int = 30) -> Optional[float]:
    """IV of the ~50-delta contract nearest `target_dte`. The standard single
    number for 'how expensive is vol on this name right now'."""
    expiry = chain.nearest_expiry(target_dte)
    if expiry is None:
        return None
    cands = [q for q in chain.quotes if q.expiry == expiry]
    if not cands:
        return None
    return min(cands, key=lambda q: abs(abs(q.delta) - 0.50)).iv


def term_structure_slope(chain: OptionChain) -> Optional[float]:
    """Back-month ATM IV minus front-month ATM IV.

    Negative = backwardation = front vol bid above back vol = the market is
    pricing near-term stress. Historically a poor time to be short premium, and
    the shape flips fast, which is exactly why it belongs in a live scanner
    rather than a static config.
    """
    front, back = atm_iv(chain, 14), atm_iv(chain, 55)
    if front is None or back is None:
        return None
    return back - front


# --------------------------------------------------------------------------- #
# Realized vol / trend from stock bars
# --------------------------------------------------------------------------- #
@dataclass
class UnderlyingStats:
    spot: float
    realized_vol_20d: Optional[float]
    realized_vol_60d: Optional[float]
    pct_from_20d_ma: Optional[float]
    drawdown_20d: Optional[float]


def underlying_stats(stock_client, symbol: str, *, lookback_days: int = 120,
                     feed=None) -> Optional[UnderlyingStats]:
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = datetime.now(timezone.utc) - timedelta(minutes=20)   # free tier: last 15m withheld
    kwargs = dict(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                  start=end - timedelta(days=lookback_days), end=end,
                  adjustment=Adjustment.RAW)
    if feed:
        kwargs["feed"] = feed
    bars = stock_client.get_stock_bars(StockBarsRequest(**kwargs)).data.get(symbol, [])
    closes = [float(b.close) for b in bars]
    if len(closes) < 25:
        return None

    def rvol(n):
        if len(closes) < n + 1:
            return None
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - n, len(closes))]
        return statistics.pstdev(rets) * math.sqrt(252) if len(rets) > 1 else None

    spot = closes[-1]
    ma20 = sum(closes[-20:]) / 20
    peak20 = max(closes[-20:])
    return UnderlyingStats(
        spot=spot,
        realized_vol_20d=rvol(20),
        realized_vol_60d=rvol(60),
        pct_from_20d_ma=(spot - ma20) / ma20,
        drawdown_20d=(spot - peak20) / peak20,
    )


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
@dataclass
class NewsBurst:
    count_24h: int
    baseline_per_day: float
    intensity: float          # count_24h / baseline; 1.0 == normal
    latest_headlines: List[str] = field(default_factory=list)


def news_burst(news_client, symbol: str, *, baseline_days: int = 30) -> Optional[NewsBurst]:
    """Headline count in the last 24h against that symbol's own 30-day baseline.

    Deliberately measures INTENSITY, not sentiment. Sentiment scoring off
    headlines is a research project with a poor track record; a burst in
    coverage is a far more robust signal that something is happening, and for a
    premium seller "something is happening" is the part that matters.
    """
    from alpaca.data.requests import NewsRequest

    now = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = now - timedelta(days=baseline_days)
    try:
        resp = news_client.get_news(NewsRequest(
            symbols=symbol, start=start, end=now, limit=1000,
            include_content=False, exclude_contentless=True))
    except Exception:
        return None

    items = resp.data.get("news", []) if hasattr(resp, "data") else list(resp)
    stamps, heads = [], []
    for n in items:
        ts = getattr(n, "created_at", None) or getattr(n, "updated_at", None)
        if ts is None:
            continue
        stamps.append(ts)
        heads.append(getattr(n, "headline", ""))
    if not stamps:
        return NewsBurst(0, 0.0, 1.0, [])

    cutoff = now - timedelta(hours=24)
    recent = [t for t in stamps if t >= cutoff]
    older = [t for t in stamps if t < cutoff]
    baseline = len(older) / max(baseline_days - 1, 1)
    intensity = (len(recent) / baseline) if baseline > 0.05 else (1.0 if not recent else 3.0)
    return NewsBurst(len(recent), baseline, intensity, heads[:5])


# --------------------------------------------------------------------------- #
# Corporate actions
# --------------------------------------------------------------------------- #
def next_ex_dividend(ca_client, symbol: str, *, horizon_days: int = 70,
                     on: Optional[date] = None) -> tuple:
    """-> (ex_date | None, source) where source is 'actual' | 'estimated' | 'none'.

    Matters for short calls: an ITM short call carrying less extrinsic than the
    dividend is an early-assignment candidate the day before ex-div, which turns
    a defined-risk spread into an unexpected short stock position.

    ALPACA HAS NO FORWARD-LOOKING DIVIDEND DATA. Verified 2026-08-12 across SPY
    and IWM over a 600-day window: every record is an already-processed
    dividend, zero future ex-dates. The API's start/end also filter on
    process_date, not ex_date, so a naive "query from today forward" returns
    nothing and the veto silently never fires -- a dead control that reads like
    a live one, which is worse than no control at all.

    So the next date is PROJECTED from the observed cadence. Index ETFs are
    metronomic about this (SPY pays the third Friday of Mar/Jun/Sep/Dec), and a
    projection accurate to a few days is enough for a veto whose question is
    "does a dividend land before this expiry". The return value says which kind
    of date you got; callers should treat 'estimated' as a real veto, because
    being wrong in the cautious direction costs one skipped trade and being
    wrong the other way costs an assignment.
    """
    from alpaca.data.enums import CorporateActionsType
    from alpaca.data.requests import CorporateActionsRequest

    today = on or date.today()
    try:
        resp = ca_client.get_corporate_actions(CorporateActionsRequest(
            symbols=[symbol], types=[CorporateActionsType.CASH_DIVIDEND],
            # Wide window and filtered in code: the API filters on process_date,
            # which lags ex_date by days to weeks depending on the issuer.
            start=today - timedelta(days=500),
            end=today + timedelta(days=horizon_days), limit=200))
    except Exception:
        return None, "none"

    ex_dates = []
    payload = getattr(resp, "data", None) or {}
    for _, actions in (payload.items() if isinstance(payload, dict) else []):
        for a in actions:
            d = getattr(a, "ex_date", None)
            if isinstance(d, datetime):
                d = d.date()
            if isinstance(d, date):
                ex_dates.append(d)
    if not ex_dates:
        return None, "none"
    ex_dates = sorted(set(ex_dates))

    future = [d for d in ex_dates if d >= today]
    if future:
        return min(future), "actual"          # declared; prefer it

    if len(ex_dates) < 2:
        return None, "none"                   # no cadence to infer from
    gaps = [(b - a).days for a, b in zip(ex_dates, ex_dates[1:])]
    gaps.sort()
    cadence = gaps[len(gaps) // 2]            # median: robust to one odd gap
    if cadence <= 0:
        return None, "none"
    projected = ex_dates[-1] + timedelta(days=cadence)
    while projected < today:
        projected += timedelta(days=cadence)
    if (projected - today).days > horizon_days:
        return None, "none"                   # outside the window we care about
    return projected, "estimated"


# --------------------------------------------------------------------------- #
# Earnings -- NOT available from Alpaca. Bring your own.
# --------------------------------------------------------------------------- #
UNKNOWN = "unknown"


class EarningsCalendar:
    """Earnings dates from a CSV you maintain, because Alpaca has no endpoint.

    CSV format (header required):

        symbol,earnings_date
        AAPL,2026-10-29
        MSFT,2026-10-27

    A symbol absent from the file reports UNKNOWN, which is NOT the same as "no
    earnings scheduled" and is deliberately not treated as such. Being short
    premium through an unexpected print is one of the fastest ways to take the
    tail loss these defined-risk structures exist to bound.

    Swap in a real provider by subclassing and overriding `next_earnings`.
    """

    def __init__(self, csv_path: Optional[str] = None):
        self.path = Path(csv_path) if csv_path else None
        self._by_symbol: Dict[str, List[date]] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open() as fh:
            for row in csv.DictReader(fh):
                sym = (row.get("symbol") or "").strip().upper()
                raw = (row.get("earnings_date") or "").strip()
                if not sym or not raw:
                    continue
                try:
                    self._by_symbol.setdefault(sym, []).append(date.fromisoformat(raw))
                except ValueError:
                    continue
        for v in self._by_symbol.values():
            v.sort()

    @property
    def loaded(self) -> bool:
        return bool(self._by_symbol)

    def next_earnings(self, symbol: str, on: Optional[date] = None):
        """-> date, or the string UNKNOWN if this calendar has no data for the
        symbol. Returns None only when the symbol IS covered and has nothing
        scheduled in the file."""
        on = on or date.today()
        if symbol.upper() not in self._by_symbol:
            return UNKNOWN
        upcoming = [d for d in self._by_symbol[symbol.upper()] if d >= on]
        return upcoming[0] if upcoming else None


# Index and broad ETFs do not report earnings. Treating them as UNKNOWN would
# suppress every suggestion on exactly the underlyings this harness targets.
NO_EARNINGS_SYMBOLS = {
    "SPY", "QQQ", "IWM", "DIA", "SPX", "XSP", "NDX", "RUT", "VIX",
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE",
    "EEM", "EFA", "TLT", "IEF", "HYG", "LQD", "GLD", "SLV", "USO", "UNG",
}
