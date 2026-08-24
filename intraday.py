"""Market intraday momentum: signal construction and evaluation.

THE STRATEGY, as published

  Gao/Han/Li/Zhou (JFE 2018)   r_ONFH = prev close -> 30 min after open
  Baltussen et al. (JFE 2021)  r_ROD  = prev close -> 30 min before close
  Zhang & Hua (Risks 2024)     r_12   = 15:00 -> 15:30

all claimed to predict r_LH, the last half hour. Enter at 15:30, exit in the
closing auction.

TWO METHODOLOGY POINTS THAT DECIDED THE RESULT HERE

1. FEED. A first pass on Alpaca's IEX feed showed r_12 at corr +0.19 (SPY) and
   +0.28 (QQQ), which looked like a real finding. IEX carries roughly 2-3% of
   consolidated volume, and re-running the identical code on SIP collapsed those
   to +0.058 and +0.072. The signal was measurement error in a thin feed. Always
   SIP here; the IEX result is not a weaker version of the truth, it is noise
   shaped like a result.

2. EXIT PRICE. The strategy exits in the closing auction, whose print is the
   DAILY bar's close -- not the 15:59 minute bar. They differ by 0.1-0.6 bp,
   which is small until you notice the whole edge is ~2.7 bp/day. Using the
   minute bar overstates or understates by up to a fifth of the thing being
   measured.

COSTS
  entry at 15:30, marketable: half-spread ~0.065 bp + slippage ~0.02 bp
  exit via MOC: crosses no spread
  SEC Section 31 fee (sell leg): 0.206 bp as of 2026-04-04
  commission at Alpaca: 0
  -> ~0.30 bp round trip, against a historical gross edge near 2.7 bp/day
"""
from __future__ import annotations

import math
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

COST_BP = 0.30


@dataclass
class Day:
    d: date
    prev_close: float
    o930: float
    c1000: float
    c1500: float
    c1530: float
    close: float          # the AUCTION print, from the daily bar

    def signal(self, name: str) -> float:
        if name == "onfh":
            return (self.c1000 - self.prev_close) / self.prev_close
        if name == "rod":
            return (self.c1530 - self.prev_close) / self.prev_close
        if name == "r12":
            return (self.c1530 - self.c1500) / self.c1500
        raise ValueError(name)

    @property
    def target(self) -> float:
        """Last half hour, 15:30 -> auction."""
        return (self.close - self.c1530) / self.c1530


def load(client, symbol: str, start: date, end: date) -> List[Day]:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    mins = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
        start=start, end=end, feed=DataFeed.SIP)).data.get(symbol, [])
    days = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
        start=start, end=end, feed=DataFeed.SIP)).data.get(symbol, [])
    daily = {str(b.timestamp)[:10]: float(b.close) for b in days}

    per: Dict[date, Dict[int, float]] = defaultdict(dict)
    for b in mins:
        et = b.timestamp - timedelta(hours=4)      # EDT
        if et.weekday() >= 5:
            continue
        per[et.date()][et.hour * 60 + et.minute] = float(b.close)

    out, prev = [], None
    for d in sorted(per):
        m = per[d]

        def at(hm, span=15):
            for k in range(hm, hm + span):
                if k in m:
                    return m[k]
            return None

        close = daily.get(str(d))
        o930, c1000, c1500, c1530 = at(9*60+30), at(10*60), at(15*60), at(15*60+30)
        if close and prev and None not in (o930, c1000, c1500, c1530):
            out.append(Day(d, prev, o930, c1000, c1500, c1530, close))
        prev = close or prev
    return out


def corr(a, b) -> float:
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def block_boot(p, nb=3000, seed=7):
    rng = random.Random(seed)
    n = len(p)
    L = max(2, int(round(n ** (1 / 3))))
    k = int(math.ceil(n / L))
    out = []
    for _ in range(nb):
        s = []
        for _ in range(k):
            i = rng.randrange(0, n - L + 1)
            s.extend(p[i:i + L])
        out.append(statistics.mean(s[:n]))
    out.sort()
    return out[int(.025 * nb)], out[int(.975 * nb)]


def evaluate(days: List[Day], name: str, cost_bp: float = COST_BP) -> dict:
    sig = [d.signal(name) for d in days]
    tgt = [d.target for d in days]
    pnl = [((1 if s > 0 else -1) * t) * 1e4 - cost_bp for s, t in zip(sig, tgt)]
    m = statistics.mean(pnl)
    sd = statistics.stdev(pnl) if len(pnl) > 1 else 0.0
    lo, hi = block_boot(pnl) if len(pnl) >= 30 else (float("nan"),) * 2
    return dict(n=len(days), corr=corr(sig, tgt), r2=corr(sig, tgt) ** 2,
                net_bp=m, sharpe=(m / sd * math.sqrt(252)) if sd else 0.0,
                t=(m / (sd / math.sqrt(len(pnl)))) if sd else 0.0,
                hit=sum(1 for x in pnl if x > 0) / len(pnl), ci=(lo, hi))
