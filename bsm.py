"""Black-Scholes-Merton pricing and greeks (European approximation).

Equity options are American-style, but for defined-risk spreads that we manage
at profit-target / stop / min-DTE rules, and for generating a synthetic chain,
the European approximation is adequate. Index options (SPX, etc.) are European.

Do NOT treat these greeks as production risk numbers. They exist so the
synthetic data source can produce a realistic chain and so strategies can
select strikes by delta the same way they would against live data.
"""
import math


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return math.inf if S > K else -math.inf
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def intrinsic(right: str, S: float, K: float) -> float:
    return max(S - K, 0.0) if right.lower() == "call" else max(K - S, 0.0)


def bs_price(S, K, T, r, sigma, right):
    """Theoretical option price. T in years."""
    right = right.lower()
    if T <= 0:
        return intrinsic(right, S, K)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if right == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S, K, T, r, sigma, right):
    """Option delta. Puts return a negative number."""
    right = right.lower()
    if T <= 0:
        if right == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, T, r, sigma)
    return norm_cdf(d1) if right == "call" else norm_cdf(d1) - 1.0


def implied_vol(price, S, K, T, r, right,
                lo=1e-4, hi=5.0, tol=1e-6, max_iter=100):
    """Invert Black-Scholes for sigma. Returns None when no solution exists.

    Needed for the Alpaca path specifically: Alpaca serves historical option
    *bars* (trade prices) but no historical greeks, so delta has to be
    reconstructed from the traded price. Bisection rather than Newton because
    vega collapses on deep-OTM short-dated contracts -- exactly the strikes
    these strategies select -- and Newton diverges there. Bisection is slower
    and does not care.

    Returns None if the price is outside the no-arbitrage band (below intrinsic
    or above the sigma=hi price). A stale or crossed print will do that, and a
    None here is a signal to drop the contract, not to clamp it.
    """
    if T <= 0 or price is None or price <= 0:
        return None
    if price < intrinsic(right, S, K) - 1e-9:
        return None            # below intrinsic: bad print, not a vol
    if price > bs_price(S, K, T, r, hi, right):
        return None            # off the top of the bracket
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = bs_price(S, K, T, r, mid, right) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    sigma = 0.5 * (lo + hi)
    # Vol is only identifiable if it actually reprices the input. On a contract
    # whose price is essentially zero, every low vol reprices to zero and the
    # bisection returns wherever it happened to stop -- a number that looks like
    # a vol and is not one. Callers select strikes by delta, so a fabricated vol
    # becomes a fabricated strike. Reject instead.
    if abs(bs_price(S, K, T, r, sigma, right) - price) > max(tol, 1e-4 * price):
        return None
    if not 1e-3 < sigma < 4.99:
        return None            # pinned to a bracket edge: not a solution
    return sigma
