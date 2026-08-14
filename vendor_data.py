"""Historical option chains from a purchased data file. Real quotes, no model.

WHY THIS EXISTS

`AlpacaDataSource` cannot observe a historical bid/ask — Alpaca serves no
as-of quote endpoint at any tier — so it reads a daily trade bar and MODELS the
spread around it. Every backtest number it produces is conditional on that
model, which is why results are reported as a grid rather than a number.

A vendor file removes the assumption entirely. The bid and ask are the ones that
existed. `spread_model` is not used, `--band` becomes unnecessary, and the
question "does this strategy work" stops being contingent on a guess.

It also fixes the sample-size problem. Alpaca's history starts February 2024,
capping the usable window at ~20 independent 45-day episodes. A vendor file
covering 2012 onward gives ~110, which is the difference between "cannot
distinguish from zero" and an actual answer.

VENDOR-NEUTRAL BY DESIGN

Every vendor names its columns differently and none of them will match. Rather
than couple to one, this maps THEIR headers onto a canonical schema via a
preset, and the presets are just dictionaries — adding a vendor is a few lines,
not a new class.

    canonical field    meaning
    ---------------    ---------------------------------------------
    quote_date         the as-of trading date
    expiration         contract expiry
    strike             strike price
    right              'call' or 'put'
    bid, ask           NBBO at the close. THE POINT OF ALL THIS.
    underlying_price   spot on quote_date
    delta, iv          optional; computed from the mid if absent
    volume, oi         optional; used for the liquidity filter

Files are indexed once into per-date shards on first use. A decade of option
quotes is tens of millions of rows and will not fit in memory, but any single
trading day comfortably does.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from bisect import bisect_left, bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from bsm import bs_delta, implied_vol
from data import OptionChain, OptionQuote, OptionsDataSource


# --------------------------------------------------------------------------- #
# Column presets. Keys are canonical; values are the vendor's header name.
# --------------------------------------------------------------------------- #
PRESETS: Dict[str, Dict[str, str]] = {
    # Cboe DataShop "Option EOD Summary" — NBBO per series, 2012 onward.
    # Use SPY rather than SPX: index bid/ask needs a separate Cboe Global
    # Indices licence starting around $1k/month, ETF options do not.
    "cboe": {
        "quote_date": "quote_date", "expiration": "expiration",
        "strike": "strike", "right": "option_type",
        "bid": "bid_eod", "ask": "ask_eod",
        "underlying_price": "underlying_bid_eod",
        "volume": "trade_volume", "oi": "open_interest",
        "delta": "delta", "iv": "implied_volatility",
    },
    "orats": {
        "quote_date": "trade_date", "expiration": "expirDate",
        "strike": "strike", "right": "__split_call_put__",
        "bid": "pBidPx", "ask": "pAskPx",
        "underlying_price": "stkPx",
        "volume": "pVolu", "oi": "pOi",
        "delta": "delta", "iv": "iv",
    },
    "ivolatility": {
        "quote_date": "date", "expiration": "expiration",
        "strike": "strike", "right": "call/put",
        "bid": "bid", "ask": "ask",
        "underlying_price": "stock_price_close",
        "volume": "volume", "oi": "open_interest",
        "delta": "delta", "iv": "iv",
    },
    # Anything already normalised to the canonical names.
    "canonical": {k: k for k in ("quote_date", "expiration", "strike", "right",
                                 "bid", "ask", "underlying_price", "volume",
                                 "oi", "delta", "iv")},
}

_RIGHT_MAP = {"c": "call", "call": "call", "p": "put", "put": "put"}


def _open(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else p.open(newline="")


def _parse_date(v: str) -> Optional[date]:
    v = (v or "").strip()
    if not v:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(v[:len(fmt) + 2].strip(), fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(v).date()
    except ValueError:
        return None


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class VendorFileDataSource(OptionsDataSource):
    """Chains from purchased CSV files. Bid/ask are real, not modelled."""

    def __init__(self, paths, underlying: str, *, preset: str = "cboe",
                 mapping: Dict[str, str] = None, index_dir: str = ".cache/vendor",
                 max_dte: int = 60, strike_band: float = 0.10,
                 min_volume: int = 0, rate: float = 0.04, verbose: bool = True):
        self.paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        self.underlying = underlying.upper()
        self.map = dict(mapping or PRESETS[preset])
        self.preset = preset
        self.index_dir = Path(index_dir) / self.underlying
        self.max_dte = max_dte
        self.strike_band = strike_band
        self.min_volume = min_volume
        self.rate = rate
        self.verbose = verbose
        self._days: List[date] = []
        self.skipped: Dict[str, int] = {}

    def _log(self, m):
        if self.verbose:
            print(f"[vendor] {m}", flush=True)

    # -- indexing --------------------------------------------------------- #
    def prepare(self, start: date = None, end: date = None) -> "VendorFileDataSource":
        """Shard the source files by quote date. Idempotent and cached."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.index_dir / "_complete.json"
        if stamp.exists():
            meta = json.loads(stamp.read_text())
            self._days = [date.fromisoformat(d) for d in meta["days"]]
            self._log(f"index cached: {len(self._days)} trading days "
                      f"{self._days[0]}..{self._days[-1]}")
            return self

        shards: Dict[str, List[dict]] = {}
        rows = kept = 0
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(path)
            self._log(f"indexing {path.name}")
            with _open(path) as fh:
                for row in csv.DictReader(fh):
                    rows += 1
                    rec = self._normalise(row)
                    if rec is None:
                        continue
                    d = rec.pop("_date")
                    if start and d < start:
                        continue
                    if end and d > end:
                        continue
                    shards.setdefault(d.isoformat(), []).append(rec)
                    kept += 1
                    if kept % 500_000 == 0:
                        self._log(f"  {kept:,} rows indexed")

        for iso, recs in shards.items():
            recs.sort(key=lambda r: (r["e"], r["k"]))
            (self.index_dir / f"{iso}.json").write_text(json.dumps(recs, separators=(",", ":")))
        self._days = sorted(date.fromisoformat(k) for k in shards)
        stamp.write_text(json.dumps({"days": [d.isoformat() for d in self._days],
                                     "rows_read": rows, "rows_kept": kept,
                                     "preset": self.preset}))
        self._log(f"indexed {kept:,} of {rows:,} rows into {len(self._days)} days")
        if not self._days:
            raise RuntimeError(
                f"no usable rows for {self.underlying}. Check the preset matches "
                f"the file's headers -- run: python vendor_data.py --inspect <file>")
        return self

    def _normalise(self, row: dict) -> Optional[dict]:
        m = self.map
        sym = (row.get("underlying_symbol") or row.get("symbol")
               or row.get("ticker") or self.underlying).strip().upper()
        if sym and sym != self.underlying:
            return None
        qd = _parse_date(row.get(m["quote_date"], ""))
        ex = _parse_date(row.get(m["expiration"], ""))
        if qd is None or ex is None:
            return None
        bid, ask = _f(row.get(m["bid"])), _f(row.get(m["ask"]))
        # A one-sided or crossed market is not tradeable. Dropping these is the
        # same liquidity discipline the Alpaca path applies, and it is the main
        # thing keeping a quote-based backtest honest.
        if bid is None or ask is None or bid <= 0 or ask <= bid:
            return None
        strike = _f(row.get(m["strike"]))
        if strike is None or strike <= 0:
            return None
        raw_right = str(row.get(m["right"], "")).strip().lower()
        right = _RIGHT_MAP.get(raw_right[:4]) or _RIGHT_MAP.get(raw_right[:1])
        if right is None:
            return None
        return {
            "_date": qd, "e": ex.isoformat(), "k": strike, "r": right,
            "b": bid, "a": ask,
            "s": _f(row.get(m.get("underlying_price", ""), "")),
            "d": _f(row.get(m.get("delta", ""), "")),
            "i": _f(row.get(m.get("iv", ""), "")),
            "v": int(_f(row.get(m.get("volume", ""), ""), 0) or 0),
        }

    # -- interface -------------------------------------------------------- #
    def trading_days(self, start: date, end: date) -> List[date]:
        return [d for d in self._days if start <= d <= end]

    def get_chain(self, underlying: str, as_of: date) -> Optional[OptionChain]:
        p = self.index_dir / f"{as_of.isoformat()}.json"
        if not p.exists():
            return None
        recs = json.loads(p.read_text())
        if not recs:
            return None

        spots = [r["s"] for r in recs if r["s"]]
        if not spots:
            self.skipped["no_spot"] = self.skipped.get("no_spot", 0) + 1
            return None
        spot = spots[0]
        lo, hi = spot * (1 - self.strike_band), spot * (1 + self.strike_band)

        quotes: List[OptionQuote] = []
        skipped = {"dte": 0, "band": 0, "thin": 0, "no_iv": 0}
        for r in recs:
            expiry = date.fromisoformat(r["e"])
            dte = (expiry - as_of).days
            if not (0 < dte <= self.max_dte):
                skipped["dte"] += 1
                continue
            if not (lo <= r["k"] <= hi):
                skipped["band"] += 1
                continue
            if r["v"] < self.min_volume:
                skipped["thin"] += 1
                continue
            mid = 0.5 * (r["b"] + r["a"])
            iv, delta = r["i"], r["d"]
            if iv is None or delta is None:
                # Vendor did not supply greeks. Derive them from the REAL mid,
                # which is still a measurement rather than a model of the spread.
                iv = implied_vol(mid, spot, r["k"], dte / 365.0, self.rate, r["r"])
                if iv is None:
                    skipped["no_iv"] += 1
                    continue
                delta = bs_delta(spot, r["k"], dte / 365.0, self.rate, iv, r["r"])
            quotes.append(OptionQuote(right=r["r"], strike=r["k"], expiry=expiry,
                                      bid=r["b"], ask=r["a"], delta=delta, iv=iv))
        for k, v in skipped.items():
            self.skipped[k] = self.skipped.get(k, 0) + v
        return OptionChain(as_of, self.underlying, spot, quotes) if quotes else None


# --------------------------------------------------------------------------- #
def inspect(path: str, rows: int = 3) -> None:
    """Print a file's headers and guess which preset fits. Run this FIRST --
    a wrong preset yields zero usable rows and a confusing error later."""
    p = Path(path)
    with _open(p) as fh:
        rdr = csv.DictReader(fh)
        headers = rdr.fieldnames or []
        sample = [next(rdr, None) for _ in range(rows)]
    print(f"{p.name}: {len(headers)} columns\n")
    for h in headers:
        print(f"  {h}")
    print("\npreset fit (higher is better):")
    for name, m in PRESETS.items():
        hit = sum(1 for v in m.values() if v in headers)
        print(f"  {name:<12} {hit}/{len(m)} columns matched")
    print("\nfirst rows:")
    for row in sample:
        if row:
            print("  " + json.dumps({k: row[k] for k in list(row)[:8]}))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a vendor options file")
    ap.add_argument("--inspect", metavar="FILE", required=True)
    a = ap.parse_args()
    inspect(a.inspect)
