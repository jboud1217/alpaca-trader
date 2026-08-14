"""Bid/ask spread modelling for the Alpaca path.

READ THIS BEFORE TRUSTING ANY REAL-DATA BACKTEST FROM THIS REPO.

The synthetic source invents bid/ask directly, so `FillModel.slippage_frac` was
the only fill assumption in the system. The Alpaca path is worse, and it is
worth being precise about why:

    Alpaca serves NO historical option quotes. The API has /options/bars,
    /options/trades, /options/quotes/LATEST and /options/snapshots. There is no
    as-of quote endpoint at any subscription tier.

So on real history we observe a *trade price* (the daily bar) and we do not
observe the spread it traded inside. That means a real-data backtest here rests
on two assumptions stacked on top of each other:

    1. bar close ~= mid            (it is not; it is wherever the last trade of
                                    the day printed, which is biased toward
                                    whichever side was lifting)
    2. half-spread ~= this model   (a guess, however well calibrated)

`FillModel.slippage_frac` then applies *on top* of an assumed spread. A result
from that pipeline is a hypothesis about profitability, not a measurement of
it. Treat the spread parameters as a sensitivity axis to sweep, never as a
number you tuned until the P&L looked good.

`calibrate_from_live_chain()` fits the parameters against real observed spreads
from a live snapshot, which is strictly better than picking them by feel. It is
still not the historical spread: it is today's liquidity, and 2024 spreads on
your underlying were probably wider.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import List, Optional


@dataclass
class SpreadModel:
    """Estimate a half-spread from a contract's price.

    Deliberately the same functional form the synthetic source uses
    (`half = max(min_half, pct_of_price * price)`) so that synthetic and real
    results stay comparable. `dte_widening` adds the one effect that form
    misses and that matters most for these strategies: near-dated cheap
    contracts quote proportionally far wider than the linear term implies.
    """

    min_half: float = 0.025       # floor, in dollars per share (half of a $0.05 wide market)
    pct_of_price: float = 0.04    # half-spread as a fraction of price
    dte_widening: float = 0.0     # extra half-spread as dte -> 0; 0.02 is a reasonable start
    max_half_frac: float = 0.5    # never let the model quote a spread wider than this * price
    use_moneyness: bool = False   # opt in to the measured moneyness curve
    use_surface: bool = False     # opt in to the full moneyness x tenor surface

    # Effective spread as a fraction of mid, by how far OTM the contract is.
    # Measured from real retail SPX fills in Beckmeyer, Branger & Gayda (2023),
    # Table 1 -- AFTER price improvement, so these are what a retail order
    # actually gets, not the quoted market.
    #
    # This matters enormously and the flat pct_of_price form misses it: a
    # far-OTM contract is CHEAP, so a percentage-of-price spread charges almost
    # nothing for it, while in reality it is the widest market in the chain.
    # For a credit spread that is precisely the leg you BUY as protection, so
    # the error runs against you on every trade and grows with wing width.
    _ES_BY_MONEYNESS = ((0.000, 0.027),   # at the money
                        (0.005, 0.070),   # 0.5-2% out of the money
                        (0.020, 0.446))   # >2% OTM -- the wing

    # Spread depends on moneyness AND tenor, and the interaction is the whole
    # story. Measured on SPY puts, 2026-08-14, n=1083 quoted contracts, median
    # quoted spread as a fraction of mid:
    #
    #                 1-7 DTE   8-21 DTE   22+ DTE
    #     ATM            0.8%       1.5%      0.8%
    #     0.5-2% OTM     3.0%       1.5%      0.9%
    #     2-5% OTM      15.4%       2.2%      0.9%
    #     >5% OTM       28.6%       6.1%      1.1%
    #
    # A >5% OTM put is 26x more expensive to trade at 3 DTE than at 30 DTE: near
    # expiry it is a near-worthless lottery ticket nobody quotes tightly, while
    # at a month out it is a real contract. Beyond ~3 weeks moneyness barely
    # matters at all -- everything quotes around 1%.
    #
    # A single-tenor curve applied across DTEs therefore gets the ranking of
    # tenors badly wrong, which is exactly the error this table exists to fix.
    _ES_SURFACE = (
        (7,   ((0.000, 0.008), (0.005, 0.030), (0.020, 0.154), (0.050, 0.286))),
        (21,  ((0.000, 0.015), (0.005, 0.015), (0.020, 0.022), (0.050, 0.061))),
        (999, ((0.000, 0.008), (0.005, 0.009), (0.020, 0.009), (0.050, 0.011))),
    )

    def _es(self, moneyness: float, dte: int) -> float:
        curve = self._ES_SURFACE[-1][1]
        for dte_hi, c in self._ES_SURFACE:
            if dte <= dte_hi:
                curve = c
                break
        m, es = max(0.0, moneyness), curve[0][1]
        for lo, val in curve:
            if m >= lo:
                es = val
            else:
                break
        return es

    def half_spread(self, price: float, dte: int = 30,
                    moneyness: float = None) -> float:
        """`moneyness` = (spot - strike)/spot for a put; larger means further OTM."""
        if moneyness is None or not self.use_moneyness:
            half = max(self.min_half, self.pct_of_price * price)
        elif self.use_surface:
            half = max(self.min_half, 0.5 * self._es(moneyness, dte) * price)
        else:
            # The tuples are LOWER bounds, so take the last bucket the contract
            # has reached -- not the first bound it falls under, which charges a
            # 1%-OTM contract the >2%-OTM rate.
            m = max(0.0, moneyness)
            es = self._ES_BY_MONEYNESS[0][1]
            for lo, val in self._ES_BY_MONEYNESS:
                if m >= lo:
                    es = val
                else:
                    break
            half = max(self.min_half, 0.5 * es * price)
        if self.dte_widening and dte < 30:
            half += self.dte_widening * (30 - max(dte, 0)) / 30.0
        return min(half, max(self.max_half_frac * price, self.min_half))

    def bid_ask(self, price: float, dte: int = 30, moneyness: float = None) -> tuple:
        half = self.half_spread(price, dte, moneyness)
        return max(0.0, price - half), price + half

    # -- presets ---------------------------------------------------------- #
    # Anchor: measured against SPY's live chain on 2026-08-10, restricted to the
    # contracts these strategies trade (|delta| 0.10-0.45, 15-60 DTE). Median
    # half-spread there was $0.02 / 0.5% of mid, p75 0.9% -- i.e. a $0.04-wide
    # market. Across the WHOLE chain the median was 1.2%, which is why the
    # calibration filters: dead far-OTM strikes quote wide and are irrelevant.
    # Re-run `--calibrate` for your own underlying; a less liquid name will not
    # look anything like this.
    # MEASURED on SPY's live chain, puts, 1-7 DTE, 2026-08-14 (spot 776.03),
    # n = 322 quoted contracts. Median quoted spread as a fraction of mid:
    #
    #     ATM (0-0.5%)     0.8%   (p25 0.6, p75 1.3)   n=20
    #     0.5-2% OTM       2.8%   (p25 1.4, p75 6.1)   n=60
    #     2-5% OTM        13.3%   (p25 5.7, p75 28.6)  n=112
    #     >5% OTM         22.2%   (p25 15.4, p75 66.7) n=130
    #
    # This is the right curve for THIS underlying at THIS tenor. It sits between
    # tight() -- which understates the OTM legs up to 4x by scaling with price --
    # and the SPX retail numbers, which overstate them ~3x because SPY is the
    # most liquid options market there is.
    #
    # Two caveats. It is QUOTED, not effective: Muravyev & Pearson (RFS 2020)
    # measure effective spreads at under 40% of quoted for traders who time
    # execution, so a limit-order strategy pays less than this. And it is ONE
    # midday snapshot in a calm tape -- these widen sharply under stress, which
    # is exactly when short gamma needs to close.
    _ES_SPY_MEASURED = ((0.000, 0.008), (0.005, 0.028),
                        (0.020, 0.133), (0.050, 0.222))

    # Per-symbol surfaces, measured 2026-08-14 14:32 ET on live chains.
    # Liquidity is NOT a property of "index ETF options" in general -- it is a
    # property of each name, and the differences are large enough to change
    # which symbols are worth trading at all.
    #
    # IWM is 2-4x worse than SPY everywhere: 3.7% vs 1.0% at the money, 6.7%
    # vs 2.3% just OTM. The cause is structural rather than incidental -- IWM
    # trades near $304 against SPY's $776, so an identically wide nickel market
    # is more than twice the percentage bite. Any low-priced underlying
    # inherits this, which is why cheap tickers make poor premium-selling
    # vehicles regardless of how liquid their SHARES are.
    _SURFACES = {
        "SPY": ((7,   ((0.000, 0.010), (0.005, 0.023), (0.020, 0.133), (0.050, 0.222))),
                (21,  ((0.000, 0.012), (0.005, 0.016), (0.020, 0.022), (0.050, 0.061))),
                (999, ((0.000, 0.011), (0.005, 0.012), (0.020, 0.009), (0.050, 0.010)))),
        "QQQ": ((7,   ((0.000, 0.012), (0.005, 0.020), (0.020, 0.049), (0.050, 0.200))),
                (21,  ((0.000, 0.011), (0.005, 0.014), (0.020, 0.019), (0.050, 0.038))),
                (999, ((0.000, 0.006), (0.005, 0.007), (0.020, 0.012), (0.050, 0.012)))),
        "IWM": ((7,   ((0.000, 0.037), (0.005, 0.067), (0.020, 0.118), (0.050, 0.286))),
                (21,  ((0.000, 0.026), (0.005, 0.032), (0.020, 0.066), (0.050, 0.133))),
                (999, ((0.000, 0.017), (0.005, 0.017), (0.020, 0.019), (0.050, 0.036)))),
    }

    @classmethod
    def for_symbol(cls, symbol: str) -> "SpreadModel":
        """Measured surface for `symbol`, falling back to SPY's for anything
        unmeasured -- which is OPTIMISTIC, since SPY is the most liquid options
        market in existence. Measure before trusting a result on a new name."""
        m = cls(min_half=0.005, pct_of_price=0.005,
                use_moneyness=True, use_surface=True)
        m._ES_SURFACE = cls._SURFACES.get(symbol.upper(), cls._SURFACES["SPY"])
        return m

    @classmethod
    def spy_measured(cls) -> "SpreadModel":
        """SPY's real quoted spread across moneyness AND tenor. This is the
        curve to use for any SPY backtest; the others are the sensitivity
        bounds around it."""
        return cls(min_half=0.005, pct_of_price=0.005,
                   use_moneyness=True, use_surface=True)

    @classmethod
    def retail_measured(cls) -> "SpreadModel":
        """Calibrated to real retail SPX execution rather than to a percentage.
        Charges 2.7% ES at the money and 44.6% on the >2% OTM wing."""
        return cls(min_half=0.01, pct_of_price=0.005, use_moneyness=True)

    @classmethod
    def tight(cls) -> "SpreadModel":
        """Megacap index ETF (SPY/QQQ) at the strikes these templates trade.
        Matches measured SPY. Tighter than `optimistic` -- and it is the
        *realistic* case for SPY, despite the name ordering."""
        return cls(min_half=0.01, pct_of_price=0.005, dte_widening=0.0)

    @classmethod
    def optimistic(cls) -> "SpreadModel":
        """Liquid but not megacap-ETF liquid."""
        return cls(min_half=0.01, pct_of_price=0.02, dte_widening=0.0)

    @classmethod
    def realistic(cls) -> "SpreadModel":
        """Liquid single-name / index ETF, typical OTM strike."""
        return cls(min_half=0.025, pct_of_price=0.04, dte_widening=0.01)

    @classmethod
    def pessimistic(cls) -> "SpreadModel":
        """Wide markets: less liquid underlying, far OTM, or thin expiry."""
        return cls(min_half=0.05, pct_of_price=0.08, dte_widening=0.03)


# ------------------------------------------------------------------------- #
# Calibration against a live snapshot
# ------------------------------------------------------------------------- #
def _parse_occ(symbol: str):
    """OCC symbol -> (expiry, right, strike). Parsed from the right because the
    root is variable length: <ROOT><YYMMDD><C|P><strike * 1000, 8 digits>."""
    from datetime import date as _date
    try:
        strike = int(symbol[-8:]) / 1000.0
        right = "call" if symbol[-9].upper() == "C" else "put"
        ymd = symbol[-15:-9]
        expiry = _date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        return expiry, right, strike
    except (ValueError, IndexError):
        return None, None, None


def calibrate_from_live_chain(client, underlying: str, *, feed=None,
                              delta_lo: float = 0.10, delta_hi: float = 0.45,
                              dte_lo: int = 15, dte_hi: int = 60,
                              as_of=None) -> dict:
    """Fit SpreadModel parameters to real observed spreads from a live chain.

    Uses the one place Alpaca *does* give you a bid/ask -- the live snapshot --
    to ground the parameters you then apply to history.

    Restricted BY DEFAULT to the contracts these strategies actually trade:
    |delta| in [0.10, 0.45] and 15-60 DTE. Calibrating across the whole chain
    would be measuring the wrong thing -- most listed contracts are dead strikes
    quoting absurdly wide, and their spreads say nothing about what a 30-delta
    45-DTE put costs to trade. The unrestricted numbers are returned alongside
    so you can see how much that filter matters.

    Fits on medians, not means: spreads have a long right tail and a mean would
    be dragged by contracts you would never touch.
    """
    from datetime import date as _date
    from alpaca.data.requests import OptionChainRequest

    as_of = as_of or _date.today()
    req = OptionChainRequest(underlying_symbol=underlying,
                             **({"feed": feed} if feed else {}))
    chain = client.get_option_chain(req)

    def usable(snap):
        q = getattr(snap, "latest_quote", None)
        if q is None or q.bid_price is None or q.ask_price is None:
            return None
        bid, ask = float(q.bid_price), float(q.ask_price)
        if bid <= 0 or ask <= bid:
            return None                   # no market, or crossed
        mid, half = 0.5 * (bid + ask), 0.5 * (ask - bid)
        if mid < 0.02:
            return None                   # sub-nickel contracts are noise
        return mid, half, half / mid

    every: List[tuple] = []
    traded: List[tuple] = []              # the delta/DTE band we actually use
    for symbol, snap in chain.items():
        o = usable(snap)
        if o is None:
            continue
        every.append(o)
        greeks = getattr(snap, "greeks", None)
        if greeks is None or greeks.delta is None:
            continue
        expiry, _, _ = _parse_occ(symbol)
        if expiry is None:
            continue
        dte = (expiry - as_of).days
        if delta_lo <= abs(float(greeks.delta)) <= delta_hi and dte_lo <= dte <= dte_hi:
            traded.append(o)

    if len(traded) < 20:
        raise RuntimeError(
            f"Only {len(traded)} usable quotes for {underlying} in the traded band "
            f"(|delta| {delta_lo}-{delta_hi}, {dte_lo}-{dte_hi} DTE) out of "
            f"{len(every)} quoted contracts. Calibrate during market hours -- "
            "outside RTH most contracts have no live market."
        )

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(int(p * len(xs)), len(xs) - 1)]

    halves = [o[1] for o in traded]
    fracs = [o[2] for o in traded]
    model = SpreadModel(
        min_half=round(pct(halves, 0.25), 4),
        pct_of_price=round(median(fracs), 4),
        dte_widening=0.0,
    )
    return {
        "model": model,
        "n_traded_band": len(traded),
        "n_all_quoted": len(every),
        "half_spread_p25_p50_p75": (pct(halves, .25), pct(halves, .50), pct(halves, .75)),
        "half_frac_p25_p50_p75": (pct(fracs, .25), pct(fracs, .50), pct(fracs, .75)),
        "all_half_frac_p25_p50_p75": (pct([o[2] for o in every], .25),
                                      pct([o[2] for o in every], .50),
                                      pct([o[2] for o in every], .75)),
        "note": ("Live snapshot -- today's liquidity, not the historical spread. "
                 "Widen before applying to older history."),
    }
