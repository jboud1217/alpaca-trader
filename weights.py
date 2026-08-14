"""Event weights and scoring for the live scanner.

HOW TO READ THE NUMBERS IN THIS FILE
------------------------------------
Every weight carries a `Provenance`. That label is the most important field in
the whole module, because it tells you which numbers are knowledge and which are
guesses wearing a decimal point:

    MEASURED    backtested in THIS repo against real data, with a pointer to
                the run that produced it. Currently: none of them.
    LITERATURE  a documented, widely replicated effect (e.g. the variance risk
                premium). The direction is well established; the weight is not.
    PRIOR       a reasoned guess. Directionally defensible, magnitude invented.

Most of the table is PRIOR. That is not a defect to be papered over -- it is the
honest starting state, and the scanner logs every suggestion with its full
factor breakdown precisely so these can be replaced by MEASURED values once
enough forward observations accumulate. A weight table that never moves off
PRIOR is a table nobody validated.

DESIGN RULES THAT KEEP THIS FROM BECOMING ASTROLOGY
---------------------------------------------------
1. A factor that cannot be computed returns None. It is DROPPED and the
   remaining weights renormalize -- never silently defaulted to zero or to a
   neutral value, which would quietly convert missing data into a mild
   endorsement. The score reports its own `coverage`.
2. Vetoes are separate from weights. Some conditions (earnings inside the
   expiry, no two-sided market) are not "negative points" to be outvoted by
   six mildly positive factors. They are disqualifying.
3. Every factor value is normalized to [-1, +1] where +1 favours SELLING
   premium and -1 says stay out, so weights are directly comparable.
4. The output is a ranked suggestion with its arithmetic attached. If you
   cannot see why a suggestion scored what it did, it is a bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Provenance(Enum):
    MEASURED = "measured"       # backtested here; cite the run
    LITERATURE = "literature"   # documented effect, direction solid, size not
    PRIOR = "prior"             # reasoned guess. unvalidated.


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class Factor:
    key: str
    weight: float
    provenance: Provenance
    rationale: str
    source: str = ""


# --------------------------------------------------------------------------- #
# THE WEIGHT TABLE -- edit here, nowhere else
# --------------------------------------------------------------------------- #
DEFAULT_FACTORS: Dict[str, Factor] = {
    "iv_minus_rv": Factor(
        key="iv_minus_rv", weight=0.30, provenance=Provenance.LITERATURE,
        rationale=(
            "Implied vol above trailing realized vol IS the variance risk premium -- "
            "the only reason short-premium strategies have a theoretical edge at all. "
            "Highest weight because it is the mechanism, not a correlate of it. "
            "Direction is well replicated; the 0.30 is a guess."),
        source="Carr & Wu (2009), 'Variance Risk Premiums'"),

    "iv_rank": Factor(
        key="iv_rank", weight=0.22, provenance=Provenance.PRIOR,
        rationale=(
            "Where current ATM IV sits in its own trailing distribution. Sells "
            "premium when it is expensive relative to this underlying's own history "
            "rather than to an absolute threshold. Needs >=60 observations before it "
            "reports anything -- the scanner accumulates them, so this factor is "
            "simply absent on a fresh install."),
        source="scanner-accumulated IV history"),

    "trend_stress": Factor(
        key="trend_stress", weight=0.16, provenance=Provenance.PRIOR,
        rationale=(
            "Put credit spreads are short downside. A name already in drawdown is "
            "where they lose. Verified concretely in this repo's own testing: the "
            "worst trade sold a 434 put with spot at 446 and exited at 417, losing "
            "$384 on $116 of credit."),
        source="probe run, this repo"),

    "term_slope": Factor(
        key="term_slope", weight=0.12, provenance=Provenance.LITERATURE,
        rationale=(
            "Back-month ATM IV minus front-month. Backwardation means the market is "
            "bidding near-term vol above longer-dated -- pricing an imminent shock. "
            "Historically a poor moment to be short gamma."),
        source="VIX term-structure literature"),

    "spread_quality": Factor(
        key="spread_quality", weight=0.12, provenance=Provenance.MEASURED,
        rationale=(
            "Half-spread as a fraction of the credit collected. This is the one "
            "factor measured directly here: SPY's traded band (|delta| 0.10-0.45, "
            "15-60 DTE) quotes a median half-spread of 0.5% of mid, while the whole "
            "chain medians 1.2%. Fill cost is the difference between the synthetic "
            "backtest making $337 and losing $305."),
        source="--calibrate on SPY, 2026-08-10; engine.py slippage sweep"),

    "news_intensity": Factor(
        key="news_intensity", weight=0.08, provenance=Provenance.PRIOR,
        rationale=(
            "24h headline count against the symbol's own 30-day baseline. Measures "
            "that something is happening, NOT what it means -- headline sentiment "
            "scoring has a poor track record and would be a false precision here. "
            "Low weight because coverage volume is a crude proxy."),
        source=""),
}


@dataclass
class ScoredFactor:
    key: str
    value: Optional[float]        # normalized [-1, +1], or None if uncomputable
    weight: float
    provenance: Provenance
    detail: str = ""


@dataclass
class Score:
    composite: float              # [-1, +1]; >0 favours selling premium
    coverage: float               # fraction of total weight actually computed
    factors: List[ScoredFactor]
    vetoes: List[str] = field(default_factory=list)

    @property
    def vetoed(self) -> bool:
        return bool(self.vetoes)

    def explain(self) -> str:
        rows = [f"  composite {self.composite:+.3f}   coverage {self.coverage:.0%}"]
        if self.vetoes:
            rows.append(f"  VETOED: {'; '.join(self.vetoes)}")
        rows.append(f"  {'factor':<16}{'value':>8}{'weight':>8}{'contrib':>9}  provenance")
        for f in sorted(self.factors, key=lambda x: -(abs(x.value or 0) * x.weight)):
            if f.value is None:
                rows.append(f"  {f.key:<16}{'--':>8}{f.weight:>8.2f}{'dropped':>9}  "
                            f"{f.provenance.value}  ({f.detail})")
            else:
                rows.append(f"  {f.key:<16}{f.value:>+8.2f}{f.weight:>8.2f}"
                            f"{f.value*f.weight:>+9.3f}  {f.provenance.value}"
                            f"{('  ('+f.detail+')') if f.detail else ''}")
        return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Normalizers: raw measurement -> [-1, +1]
# --------------------------------------------------------------------------- #
def norm_iv_minus_rv(iv: Optional[float], rv: Optional[float]):
    """VRP as a fraction of realized vol. +1 at IV running 35% above RV."""
    if iv is None or rv is None or rv <= 0.01:
        return None, "no iv/rv"
    ratio = (iv - rv) / rv
    return clamp(ratio / 0.35), f"iv={iv:.1%} rv={rv:.1%} ratio={ratio:+.0%}"


def norm_iv_rank(pct: Optional[float], n_obs: int = 0):
    """Percentile in trailing history -> [-1, +1]."""
    if pct is None:
        return None, f"needs >=60 obs, have {n_obs}"
    return clamp(2.0 * pct - 1.0), f"pctile={pct:.0%} n={n_obs}"


def norm_term_slope(slope: Optional[float]):
    """Back minus front ATM IV. +1 at 4 vol points of contango."""
    if slope is None:
        return None, "no term structure"
    return clamp(slope / 0.04), f"back-front={slope:+.3f}"


def norm_trend_stress(drawdown_20d: Optional[float], pct_from_ma: Optional[float]):
    """Drawdown from the 20-day high, nudged by distance from the 20-day MA."""
    if drawdown_20d is None:
        return None, "no price history"
    v = clamp(drawdown_20d / 0.04 + 0.5)
    if pct_from_ma is not None:
        v = clamp(0.75 * v + 0.25 * clamp(pct_from_ma / 0.03))
    return v, f"dd20={drawdown_20d:+.1%} vs_ma={0.0 if pct_from_ma is None else pct_from_ma:+.1%}"


# Above this, round-trip fill cost is a multiple of the strategy's entire
# measured expectancy and the trade cannot pay for itself. See veto note below.
MAX_SPREAD_FRAC_OF_CREDIT = 0.15


def norm_spread_quality(half_spread: Optional[float], credit: Optional[float]):
    """Half-spread as a fraction of credit collected. +1 when negligible, -1
    once round-trip fill cost eats ~30% of the credit."""
    if half_spread is None or credit is None or credit <= 0:
        return None, "no credit/spread"
    frac = half_spread / credit
    return clamp(1.0 - frac / 0.15), f"halfspread/credit={frac:.1%}"


def spread_veto(half_spread: Optional[float], credit: Optional[float]) -> Optional[str]:
    """Fill cost is a VETO, not merely a negative weight.

    Learned from a live scan on 2026-08-10: IWM ranked top of the board at
    composite +0.313 while quoting a round-trip spread of 24% of the credit,
    because a strong VRP reading and a steep term slope simply outvoted the
    -0.07 that fill quality contributed. But this repo's own slippage sweep puts
    these templates' expectancy at roughly 5% of credit -- so a 24% fill cost is
    several times the entire edge. No combination of favourable events makes a
    trade you cannot get out of worth taking. Weights rank tradeable candidates;
    they must not be able to vote a untradeable one onto the board.
    """
    if half_spread is None or credit is None or credit <= 0:
        return None
    frac = half_spread / credit
    if frac > MAX_SPREAD_FRAC_OF_CREDIT:
        return (f"round-trip spread {frac:.0%} of credit exceeds "
                f"{MAX_SPREAD_FRAC_OF_CREDIT:.0%} -- fill cost swamps expectancy")
    return None


def norm_news_intensity(intensity: Optional[float]):
    """1.0 == normal coverage. Elevated coverage counts against."""
    if intensity is None:
        return None, "no news data"
    return clamp(-(intensity - 1.0) / 2.0), f"intensity={intensity:.1f}x"


# --------------------------------------------------------------------------- #
def score(values: Dict[str, tuple], vetoes: List[str],
          factors: Dict[str, Factor] = None) -> Score:
    """Combine normalized factor values into a composite.

    `values` maps factor key -> (normalized value or None, detail string).
    Uncomputable factors drop out and the rest renormalize, so a score built
    from two factors is not silently compared against one built from six --
    `coverage` is what tells you which you are looking at.
    """
    factors = factors or DEFAULT_FACTORS
    scored, num, den, total = [], 0.0, 0.0, 0.0
    for key, f in factors.items():
        total += f.weight
        val, detail = values.get(key, (None, "not evaluated"))
        scored.append(ScoredFactor(key, val, f.weight, f.provenance, detail))
        if val is not None:
            num += val * f.weight
            den += f.weight
    composite = (num / den) if den > 0 else 0.0
    return Score(composite=composite, coverage=(den / total if total else 0.0),
                 factors=scored, vetoes=list(vetoes))


def provenance_report(factors: Dict[str, Factor] = None) -> str:
    factors = factors or DEFAULT_FACTORS
    by = {}
    for f in factors.values():
        by.setdefault(f.provenance, []).append(f)
    out = ["Weight table provenance", "=" * 72]
    for p in (Provenance.MEASURED, Provenance.LITERATURE, Provenance.PRIOR):
        fs = by.get(p, [])
        share = sum(f.weight for f in fs)
        out.append(f"\n{p.value.upper()}  ({len(fs)} factors, {share:.0%} of total weight)")
        for f in fs:
            out.append(f"  {f.key}  (w={f.weight})")
            out.append(f"      {f.rationale}")
            if f.source:
                out.append(f"      source: {f.source}")
    unmeasured = sum(f.weight for f in factors.values()
                     if f.provenance is not Provenance.MEASURED)
    out.append("\n" + "=" * 72)
    out.append(f"{unmeasured:.0%} of total weight is NOT measured against real outcomes.")
    out.append("Treat composite scores as a ranking heuristic, not a probability.")
    return "\n".join(out)
