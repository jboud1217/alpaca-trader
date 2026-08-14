"""Data layer.

Two sources implement the same OptionsDataSource interface:

  * SyntheticDataSource  - fully offline. Generates a GBM price path and prices
    a Black-Scholes chain on top of it. Used to validate the ENGINE with no
    network or API keys. IMPORTANT: it bakes a variance-risk-premium edge into
    the data by construction (implied vol is set above realized vol), so a
    profitable premium-selling result here proves the plumbing works, NOT that
    the strategy has a real edge. Never confuse the two.

  * AlpacaDataSource     - real historical options data via alpaca-py. This is
    where you actually test whether an edge exists. Implemented against
    alpaca-py 0.43.x. Requires network + keys, and carries one caveat big
    enough to belong up here: Alpaca has NO historical option quotes endpoint,
    so the bid/ask is MODELLED (see spread.py), not observed. Read that class's
    docstring before believing any number it produces.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from datetime import date, datetime, time as dtime, timedelta
from typing import List, Optional

from bsm import bs_delta, bs_price, implied_vol, intrinsic


@dataclass
class OptionQuote:
    right: str
    strike: float
    expiry: date
    bid: float
    ask: float
    delta: float
    iv: float
    # OCC contract symbol. Populated only by the LIVE chain (events.live_chain),
    # because only there does a real tradeable contract exist. The historical
    # and synthetic sources leave it None, and execution.py refuses to build an
    # order without it -- that is deliberate: it makes it structurally
    # impossible to submit an order derived from backtest or simulated data.
    symbol: Optional[str] = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class OptionChain:
    as_of: date
    underlying: str
    spot: float
    quotes: List[OptionQuote]

    def expiries(self) -> List[date]:
        return sorted({q.expiry for q in self.quotes})

    def nearest_expiry(self, target_dte: int) -> Optional[date]:
        exps = self.expiries()
        if not exps:
            return None
        target = self.as_of + timedelta(days=target_dte)
        return min(exps, key=lambda e: abs((e - target).days))

    def by_expiry(self, expiry: date, right: str) -> List[OptionQuote]:
        return sorted(
            (q for q in self.quotes if q.expiry == expiry and q.right == right),
            key=lambda q: q.strike,
        )

    def select_by_delta(self, expiry, right, target_delta) -> Optional[OptionQuote]:
        cands = self.by_expiry(expiry, right)
        if not cands:
            return None
        return min(cands, key=lambda q: abs(abs(q.delta) - abs(target_delta)))

    def leg_at_offset(self, expiry, right, anchor_strike, width) -> Optional[OptionQuote]:
        """Contract ~`width` OTM beyond `anchor_strike` (the long wing)."""
        target = anchor_strike - width if right == "put" else anchor_strike + width
        cands = self.by_expiry(expiry, right)
        if not cands:
            return None
        return min(cands, key=lambda q: abs(q.strike - target))

    def find(self, right, strike, expiry) -> Optional[OptionQuote]:
        for q in self.quotes:
            if q.right == right and q.expiry == expiry and abs(q.strike - strike) < 1e-6:
                return q
        return None


class OptionsDataSource:
    """Interface. Implement these two methods for any data provider."""

    def trading_days(self, start: date, end: date) -> List[date]:
        raise NotImplementedError

    def get_chain(self, underlying: str, as_of: date) -> Optional[OptionChain]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Synthetic source
# --------------------------------------------------------------------------- #
class SyntheticDataSource(OptionsDataSource):
    def __init__(
        self,
        start: date,
        end: date,
        s0: float = 100.0,
        realized_vol: float = 0.20,
        vrp: float = 0.03,          # IV = realized_vol + vrp (the baked-in edge)
        put_skew: float = 0.02,     # extra IV added to puts, linear in moneyness
        drift: float = 0.05,
        rate: float = 0.04,
        seed: int = 7,
        strike_step: float = 1.0,
        strikes_each_side: int = 40,
        expiry_step_days: int = 7,
        max_dte: int = 60,
        min_half_spread: float = 0.03,
        spread_pct: float = 0.03,   # half-spread as pct of price
    ):
        self.underlying_symbol = "SYN"
        self.realized_vol = realized_vol
        self.vrp = vrp
        self.put_skew = put_skew
        self.rate = rate
        self.strike_step = strike_step
        self.strikes_each_side = strikes_each_side
        self.expiry_step_days = expiry_step_days
        self.max_dte = max_dte
        self.min_half_spread = min_half_spread
        self.spread_pct = spread_pct

        self._days = self._business_days(start, end)
        self._path = self._simulate_path(s0, drift, realized_vol, seed)
        # Real options expire on FIXED calendar dates, not "N days from today".
        # Use weekly Friday expirations spanning the whole window (+ max_dte tail)
        # so a position opened today is still findable/valuable on later days.
        self._expiries = self._weekly_expiries(start, end + timedelta(days=max_dte))

    @staticmethod
    def _business_days(start, end) -> List[date]:
        days, d = [], start
        while d <= end:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        return days

    @staticmethod
    def _weekly_expiries(start, end) -> List[date]:
        d = start
        while d.weekday() != 4:  # advance to first Friday
            d += timedelta(days=1)
        exps = []
        while d <= end:
            exps.append(d)
            d += timedelta(days=7)
        return exps

    def _simulate_path(self, s0, drift, vol, seed) -> dict:
        rng = random.Random(seed)
        dt = 1.0 / 252.0
        path, s = {}, s0
        for d in self._days:
            z = rng.gauss(0.0, 1.0)
            s *= math.exp((drift - 0.5 * vol * vol) * dt + vol * math.sqrt(dt) * z)
            path[d] = s
        return path

    def trading_days(self, start, end) -> List[date]:
        return [d for d in self._days if start <= d <= end]

    def _iv_for(self, right, strike, spot) -> float:
        iv = self.realized_vol + self.vrp
        if right == "put":
            moneyness = max(0.0, (spot - strike) / spot)  # deeper OTM puts pricier
            iv += self.put_skew * moneyness * 10.0
        return iv

    def get_chain(self, underlying, as_of) -> Optional[OptionChain]:
        spot = self._path.get(as_of)
        if spot is None:
            return None
        atm = round(spot / self.strike_step) * self.strike_step
        expiries = [
            e for e in self._expiries
            if 0 < (e - as_of).days <= self.max_dte
        ]
        quotes = []
        for expiry in expiries:
            T = max((expiry - as_of).days, 0) / 365.0
            for k in range(-self.strikes_each_side, self.strikes_each_side + 1):
                strike = atm + k * self.strike_step
                if strike <= 0:
                    continue
                for right in ("put", "call"):
                    iv = self._iv_for(right, strike, spot)
                    price = bs_price(spot, strike, T, self.rate, iv, right)
                    if price < 0.01:
                        continue
                    delta = bs_delta(spot, strike, T, self.rate, iv, right)
                    half = max(self.min_half_spread, self.spread_pct * price)
                    quotes.append(
                        OptionQuote(
                            right=right, strike=strike, expiry=expiry,
                            bid=max(0.0, price - half), ask=price + half,
                            delta=delta, iv=iv,
                        )
                    )
        return OptionChain(as_of, underlying, spot, quotes)


# --------------------------------------------------------------------------- #
# Alpaca source (real; scaffold — verify against your alpaca-py version)
# --------------------------------------------------------------------------- #
OPTIONS_HISTORY_FLOOR = date(2024, 2, 1)   # Alpaca serves no options history before this


class AlpacaDataSource(OptionsDataSource):
    """Historical options data via alpaca-py, reconstructed as-of each date.

    THE CENTRAL CAVEAT
    ------------------
    Alpaca serves no historical option quotes. The endpoints are /options/bars,
    /options/trades, /options/quotes/LATEST and /options/snapshots -- the last
    two are live-only at every subscription tier. So this source cannot observe
    the historical bid/ask. It observes a daily *trade* bar and models the
    spread around it (see spread.py). Two assumptions therefore sit underneath
    every number this produces:

        bar close ~= mid          (biased toward whichever side was lifting)
        half-spread ~= SpreadModel  (calibrated at best, still a guess)

    Sweep both. A real-data result from this harness is a hypothesis, not a
    measurement. `run_backtest.py --real --band` exists to make that visible.

    OTHER CONSTRAINTS THAT BITE
    ---------------------------
    * History floor: February 2024. Anything earlier returns nothing.
    * Free tier is the *indicative* feed (a 15-min-delayed derivative of OPRA),
      not the consolidated BBO. Real OPRA is a paid subscription.
    * Survivorship: `get_option_contracts` reports a contract's status *now*.
      Expired contracts go inactive, so both statuses must be queried or the
      whole past disappears.
    * Liquidity: a contract with no bar on a date did not trade that date. That
      is treated as untradeable rather than filled at theoretical value, which
      is the single biggest thing keeping this honest.

    DESIGN
    ------
    Bulk-load the whole window once (spot series, contract universe, all daily
    bars), cache to disk, then serve each as-of chain from memory. Fetching
    per-day would need thousands of calls against a 200/min limit; this needs
    tens, and only on the first run.
    """

    def __init__(self, api_key: str, secret_key: str, *,
                 max_dte: int = 60,
                 strike_band: float = 0.30,
                 rate: float = 0.04,
                 spread_model=None,
                 price_field: str = "close",
                 min_volume: int = 1,
                 min_price: float = 0.05,
                 cache_dir: str = ".cache/alpaca",
                 requests_per_min: int = 180,
                 verbose: bool = True):
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        from cache import DiskCache, RateLimiter
        from spread import SpreadModel

        self.opt = OptionHistoricalDataClient(api_key, secret_key)
        self.stock = StockHistoricalDataClient(api_key, secret_key)
        self.trading = TradingClient(api_key, secret_key, paper=True)

        self.max_dte = max_dte
        self.strike_band = strike_band          # fraction of spot to keep either side
        self.rate = rate
        self.spread_model = spread_model or SpreadModel.realistic()
        self.price_field = price_field          # 'close' (decision-time) or 'vwap' (smoother)
        self.min_volume = min_volume
        # Below a few cents an option carries no recoverable vol: every low sigma
        # reprices to ~0, so the solver returns a number that looks like an IV and
        # is not one. Strikes get picked by delta, so that fabricated IV would
        # become a fabricated strike. Also nobody sells $0.03 of premium.
        self.min_price = min_price
        self.verbose = verbose

        self.cache = DiskCache(cache_dir)
        self.limiter = RateLimiter(requests_per_min)

        self._prepared_key = None
        self._spot: dict = {}                   # date -> close
        self._universe: list = []               # contract dicts
        self._bars: dict = {}                   # (symbol, isodate) -> price dict
        self.skipped: dict = {}                 # diagnostic counters
        # as_of -> reconstructed (right, strike, expiry, price, dte, delta, iv).
        # Everything expensive -- the implied-vol solve and the greeks -- depends
        # only on the price bar, NOT on the spread model or slippage. A --band
        # sweep re-serves the same 375 days across 24 assumption cells, so
        # without this the IV solver runs ~1.8 billion times for 24 identical
        # answers. Cache the reconstruction; re-apply only the spread at serve.
        self._chain_cache: dict = {}

    # -- logging ----------------------------------------------------------- #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[alpaca] {msg}", flush=True)

    # -- bulk load --------------------------------------------------------- #
    def prepare(self, underlying: str, start: date, end: date) -> "AlpacaDataSource":
        """Download and cache everything the window needs. Idempotent."""
        key = f"{underlying}|{start}|{end}|{self.max_dte}|{self.strike_band}"
        if self._prepared_key == key:
            return self
        if start < OPTIONS_HISTORY_FLOOR:
            raise ValueError(
                f"start={start} precedes Alpaca's options history floor "
                f"({OPTIONS_HISTORY_FLOOR}). Nothing exists before then."
            )

        self._spot = self._load_spot(underlying, start, end)
        if not self._spot:
            raise RuntimeError(f"No underlying bars for {underlying} in {start}..{end}.")
        lo_spot, hi_spot = min(self._spot.values()), max(self._spot.values())
        self._log(f"spot {underlying}: {len(self._spot)} days, "
                  f"range {lo_spot:.2f}..{hi_spot:.2f}")

        strike_lo = lo_spot * (1 - self.strike_band)
        strike_hi = hi_spot * (1 + self.strike_band)
        self._universe = self._load_universe(
            underlying, start, end + timedelta(days=self.max_dte), strike_lo, strike_hi)
        self._log(f"contracts in window: {len(self._universe)} "
                  f"(strikes {strike_lo:.0f}..{strike_hi:.0f})")
        if not self._universe:
            raise RuntimeError(
                f"No option contracts found for {underlying}. Check the symbol has "
                "listed options and that your account has options data enabled."
            )

        symbols = [c["symbol"] for c in self._universe]
        self._bars = self._load_bars(symbols, start, end)
        self._log(f"contract-days with a trade: {len(self._bars)}")

        self._build_index()
        self._prepared_key = key
        return self

    def _build_index(self) -> None:
        """Index the universe by expiry, with dates parsed and strikes sorted.

        Without this, get_chain linearly scans every contract and re-parses its
        expiry string on every single day. Over a 375-day window with 117k
        contracts and a 24-cell assumption grid that is tens of millions of
        redundant date parses, and the band sweep never finishes. Expiries are
        pre-parsed once and strikes kept sorted so each day bisects into just
        the band it needs.
        """
        by_expiry: dict = {}
        for c in self._universe:
            by_expiry.setdefault(date.fromisoformat(c["expiry"]), []).append(
                (c["strike"], c["symbol"], c["right"]))
        for v in by_expiry.values():
            v.sort()                       # by strike, for bisect in get_chain
        self._by_expiry = by_expiry
        self._expiries = sorted(by_expiry)
        self._strikes_by_expiry = {e: [row[0] for row in v] for e, v in by_expiry.items()}

    def _load_spot(self, underlying: str, start: date, end: date) -> dict:
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        def fetch():
            self.limiter.wait()
            # RAW, not adjusted: option strikes are in unadjusted terms, so a
            # split-adjusted underlying would silently misalign every moneyness
            # calculation against the strikes it is compared to.
            req = StockBarsRequest(
                symbol_or_symbols=underlying, timeframe=TimeFrame.Day,
                start=datetime.combine(start, dtime.min),
                end=datetime.combine(end, dtime.max),
                adjustment=Adjustment.RAW,
            )
            bars = self.stock.get_stock_bars(req)
            data = bars.data.get(underlying, [])
            return {b.timestamp.date().isoformat(): float(b.close) for b in data}

        raw = self.cache.get_or_fetch("spot", f"{underlying}|{start}|{end}", fetch)
        return {date.fromisoformat(k): v for k, v in raw.items()}

    def _load_universe(self, underlying, exp_start, exp_end, strike_lo, strike_hi) -> list:
        from alpaca.trading.enums import AssetStatus
        from alpaca.trading.requests import GetOptionContractsRequest

        def fetch():
            out, seen = [], set()
            # Both statuses: `status` describes the contract TODAY, so every
            # contract that already expired reads as inactive. Querying only
            # active silently deletes the entire past.
            for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
                token, pages = None, 0
                while True:
                    self.limiter.wait()
                    req = GetOptionContractsRequest(
                        underlying_symbols=[underlying],
                        status=status,
                        expiration_date_gte=exp_start,
                        expiration_date_lte=exp_end,
                        strike_price_gte=str(round(strike_lo, 2)),
                        strike_price_lte=str(round(strike_hi, 2)),
                        limit=10_000,
                        page_token=token,
                    )
                    resp = self.trading.get_option_contracts(req)
                    batch = resp.option_contracts or []
                    for c in batch:
                        if c.symbol in seen:
                            continue
                        seen.add(c.symbol)
                        out.append({
                            "symbol": c.symbol,
                            "strike": float(c.strike_price),
                            "expiry": c.expiration_date.isoformat(),
                            "right": "call" if c.type.value == "call" else "put",
                        })
                    pages += 1
                    token = getattr(resp, "next_page_token", None)
                    if not token or not batch:
                        break
                    if pages > 200:
                        self._log("WARNING: contract pagination hit 200 pages; truncating")
                        break
            return out

        key = f"{underlying}|{exp_start}|{exp_end}|{strike_lo:.2f}|{strike_hi:.2f}"
        return self.cache.get_or_fetch("universe", key, fetch)

    def _load_bars(self, symbols: list, start: date, end: date) -> dict:
        from alpaca.data.requests import OptionBarsRequest
        from alpaca.data.timeframe import TimeFrame

        chunk_size = 100        # keep the querystring well under any URL limit
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        out: dict = {}
        for i, chunk in enumerate(chunks, 1):
            key = f"{start}|{end}|{','.join(sorted(chunk))}"

            def fetch(chunk=chunk):
                self.limiter.wait()
                req = OptionBarsRequest(
                    symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                    start=datetime.combine(start, dtime.min),
                    end=datetime.combine(end, dtime.max),
                )
                bars = self.opt.get_option_bars(req)   # SDK paginates internally
                rows = {}
                for sym, series in bars.data.items():
                    for b in series:
                        rows[f"{sym}|{b.timestamp.date().isoformat()}"] = {
                            "close": float(b.close),
                            "vwap": float(b.vwap) if b.vwap else float(b.close),
                            "volume": int(b.volume or 0),
                        }
                return rows

            rows = self.cache.get_or_fetch("bars", key, fetch)
            out.update(rows)
            if self.verbose and (i % 10 == 0 or i == len(chunks)):
                self._log(f"bars: chunk {i}/{len(chunks)}")
        return out

    # -- interface --------------------------------------------------------- #
    def trading_days(self, start: date, end: date) -> List[date]:
        from alpaca.trading.requests import GetCalendarRequest

        def fetch():
            self.limiter.wait()
            cal = self.trading.get_calendar(GetCalendarRequest(start=start, end=end))
            return [c.date.isoformat() for c in cal]

        raw = self.cache.get_or_fetch("calendar", f"{start}|{end}", fetch)
        return [date.fromisoformat(d) for d in raw]

    def get_chain(self, underlying: str, as_of: date) -> Optional[OptionChain]:
        if self._prepared_key is None:
            raise RuntimeError("Call prepare(underlying, start, end) before get_chain().")

        spot = self._spot.get(as_of)
        if spot is None:
            return None                      # non-trading day or missing bar

        cached = self._chain_cache.get(as_of)
        if cached is not None:
            if not cached:
                return None
            quotes = []
            for right, strike, expiry, price, dte, delta, iv in cached:
                # (spot - strike)/spot for a put; the sign flips for calls so
                # that "further OTM" is positive in both cases.
                mny = ((spot - strike) / spot) if right == "put" else ((strike - spot) / spot)
                bid, ask = self.spread_model.bid_ask(price, dte, mny)
                quotes.append(OptionQuote(right=right, strike=strike, expiry=expiry,
                                          bid=bid, ask=ask, delta=delta, iv=iv))
            return OptionChain(as_of, underlying, spot, quotes)

        lo, hi = spot * (1 - self.strike_band), spot * (1 + self.strike_band)
        quotes: List[OptionQuote] = []
        recon: List[tuple] = []
        skipped = {"no_bar": 0, "thin": 0, "too_cheap": 0,
                   "deep_itm_no_timevalue": 0, "no_iv": 0}

        as_of_iso = as_of.isoformat()
        for expiry in self._expiries:
            dte = (expiry - as_of).days
            if dte <= 0:
                continue
            if dte > self.max_dte:
                break                        # expiries are sorted: nothing later qualifies

            strikes = self._strikes_by_expiry[expiry]
            rows = self._by_expiry[expiry]
            i = bisect_left(strikes, lo)
            j = bisect_right(strikes, hi)
            T = dte / 365.0

            for strike, symbol, right in rows[i:j]:
                bar = self._bars.get(f"{symbol}|{as_of_iso}")
                if bar is None:
                    skipped["no_bar"] += 1   # did not trade: untradeable, not free
                    continue
                if bar["volume"] < self.min_volume:
                    skipped["thin"] += 1
                    continue

                price = bar[self.price_field]
                if price < self.min_price:
                    skipped["too_cheap"] += 1
                    continue
                iv = implied_vol(price, spot, strike, T, self.rate, right)
                if iv is None:
                    # Two very different causes, worth separating in diagnostics.
                    # Deep-ITM contracts have so little time value that a price
                    # rounded to the cent lands below the zero-vol floor --
                    # benign, they are unusable and untraded anyway. Anything
                    # else is a genuinely odd print and worth knowing about.
                    if price <= intrinsic(right, spot, strike) + 0.02:
                        skipped["deep_itm_no_timevalue"] += 1
                    else:
                        skipped["no_iv"] += 1
                    continue

                delta = bs_delta(spot, strike, T, self.rate, iv, right)
                # (spot - strike)/spot for a put; the sign flips for calls so
                # that "further OTM" is positive in both cases.
                mny = ((spot - strike) / spot) if right == "put" else ((strike - spot) / spot)
                bid, ask = self.spread_model.bid_ask(price, dte, mny)
                recon.append((right, strike, expiry, price, dte, delta, iv))
                quotes.append(OptionQuote(
                    right=right, strike=strike, expiry=expiry,
                    bid=bid, ask=ask, delta=delta, iv=iv,
                ))

        for k, v in skipped.items():
            self.skipped[k] = self.skipped.get(k, 0) + v
        self._chain_cache[as_of] = recon
        if not quotes:
            return None
        return OptionChain(as_of, underlying, spot, quotes)
