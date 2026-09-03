"""Read-only account/system snapshot for the phone dashboard.

Served behind a Lambda Function URL because the dashboard is a static HTML file
and cannot talk to Alpaca directly: Alpaca sends no CORS headers, and putting
API keys in a page you carry around would hand trading credentials to anyone who
picked up the file.

THIS HANDLER CANNOT TRADE.
It never constructs an Executor, never imports an order request type, and never
touches storage.PENDING except to read. That is deliberate -- the endpoint is
public (auth is a bearer token, not IAM), so it is written on the assumption
that the token will eventually leak. The worst case then is that somebody reads
your P&L, not that they move your money.

PERFORMANCE MATH -- the part that is easy to get wrong:
Alpaca's portfolio-history response carries a `profit_loss` array, and summing
it looks obvious and is wrong. Each entry is that point's P&L relative to
`base_value`, not an increment, so a sum double-counts. Measured on this account
it produced -$900,009 for a week in which the true change was -$9.24. Period
P&L is `equity[-1] - base_value`; nothing else.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, "/var/task")

import boto3

import dynamo_store
import execution as ex
import storage

_PREFIX = os.environ.get("PARAM_PREFIX", "/alpaca/dev")
_ssm_client = None


def _ssm():
    global _ssm_client
    if _ssm_client is None:
        _ssm_client = boto3.client("ssm")
    return _ssm_client


def _params() -> dict:
    out, token = {}, None
    while True:
        kw = {"Path": _PREFIX, "WithDecryption": True, "MaxResults": 10}
        if token:
            kw["NextToken"] = token
        resp = _ssm().get_parameters_by_path(**kw)
        for p in resp.get("Parameters", []):
            out[p["Name"].rsplit("/", 1)[-1]] = p["Value"]
        token = resp.get("NextToken")
        if not token:
            break
    return out


def _flag(cfg, name) -> bool:
    return str(cfg.get(name, "")).strip().lower() in ("on", "true", "1", "yes")


def _alpaca_get(cfg, path: str, paper: bool):
    import urllib.request
    base = ("https://paper-api.alpaca.markets" if paper
            else "https://api.alpaca.markets")
    req = urllib.request.Request(base + path, headers={
        "APCA-API-KEY-ID": cfg["alpaca_key"],
        "APCA-API-SECRET-KEY": cfg["alpaca_secret"]})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _period(cfg, paper, label, period, timeframe) -> dict:
    """P&L over one window. See the module docstring on why this is not a sum."""
    try:
        h = _alpaca_get(cfg, f"/v2/account/portfolio/history"
                             f"?period={period}&timeframe={timeframe}", paper)
    except Exception as e:
        return {"label": label, "error": f"{type(e).__name__}"}
    eq = [e for e in (h.get("equity") or []) if e]
    base = h.get("base_value")
    if not eq or base in (None, 0):
        return {"label": label, "pl": 0.0, "pct": 0.0, "points": len(eq)}
    pl = eq[-1] - base
    return {"label": label, "pl": round(pl, 2), "pct": round(100.0 * pl / base, 3),
            "base": round(base, 2), "equity": round(eq[-1], 2),
            "series": [round(x, 2) for x in eq[-120:]], "points": len(eq)}


def handler(event, context):
    # ---- auth --------------------------------------------------------- #
    cfg = _params()
    want = cfg.get("stats_token")
    hdrs = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    got = (hdrs.get("authorization") or "").replace("Bearer ", "").strip()
    qs = (event.get("queryStringParameters") or {}).get("t", "")
    if not want or (got != want and qs != want):
        return {"statusCode": 401,
                "headers": {"content-type": "application/json"},
                "body": json.dumps({"error": "unauthorized"})}

    live = _flag(cfg, "live_money")
    paper = not live
    dynamo_store.install()
    halt = dynamo_store.day_halt()

    # Every cap, not just three. Reporting default ceilings while the system
    # enforces configured ones makes the dashboard actively misleading -- the
    # risk meters would show $2,500 of headroom against an enforced $550.
    def _num(key, default, cast=float):
        try:
            return cast(cfg.get(key, default))
        except (TypeError, ValueError):
            return cast(default)
    limits = ex.RiskLimits(
        account_equity=_num("equity", 25000),
        risk_per_trade_frac=_num("risk_frac", 0.02),
        max_contracts=_num("max_contracts", 2, int),
        max_risk_per_trade=_num("max_risk_per_trade", 750),
        max_open_risk=_num("max_open_risk", 2500),
        max_orders_per_day=ex.opt_int(cfg.get("max_orders_per_day")),
        max_new_risk_per_day=_num("max_new_risk_per_day", 1000))

    out = {"ts": datetime.now(timezone.utc).isoformat(),
           "env": _PREFIX.rsplit("/", 1)[-1]}

    # ---- system posture ------------------------------------------------ #
    try:
        kill_on = storage.KILL.engaged()
    except Exception:
        kill_on = True
    out["system"] = {
        "armed": _flag(cfg, "armed"), "live": live,
        "halted": halt.engaged(), "blocked": kill_on,
        "blocked_reason": storage.KILL.describe() if kill_on else "",
        "symbols": cfg.get("symbols", ""),
        "threshold": float(cfg.get("threshold", 0.15)),
        "strategy": (f"{cfg.get('wing_width','?')}-wide put credit spread, "
                     f"{cfg.get('target_dte','?')} DTE, "
                     f"{cfg.get('short_delta','?')} delta"),
        "exits": (f"take {float(cfg.get('profit_take',0.5)):.0%} / "
                  f"stop {cfg.get('stop_mult','?')}x / "
                  f"roll {cfg.get('min_dte','?')} DTE"),
    }

    # ---- account + performance ----------------------------------------- #
    try:
        a = _alpaca_get(cfg, "/v2/account", paper)
        out["account"] = {
            "number": a.get("account_number"), "status": a.get("status"),
            "equity": float(a.get("equity", 0)),
            "last_equity": float(a.get("last_equity", 0)),
            "cash": float(a.get("cash", 0)),
            "buying_power": float(a.get("buying_power", 0)),
            "mode": "LIVE" if live else "paper",
        }
    except Exception as e:
        out["account"] = {"error": f"{type(e).__name__}: {e}"}

    out["performance"] = [
        _period(cfg, paper, "Today", "1D", "5Min"),
        _period(cfg, paper, "Week", "1W", "1H"),
        _period(cfg, paper, "Month", "1M", "1D"),
        _period(cfg, paper, "Year", "1A", "1D"),
    ]

    # ---- open positions ------------------------------------------------ #
    try:
        pos = _alpaca_get(cfg, "/v2/positions", paper)
        out["positions"] = [{
            "symbol": p.get("symbol"),
            "qty": float(p.get("qty", 0)),
            "market_value": float(p.get("market_value") or 0),
            "cost_basis": float(p.get("cost_basis") or 0),
            "unrealized_pl": float(p.get("unrealized_pl") or 0),
            "unrealized_plpc": float(p.get("unrealized_plpc") or 0) * 100,
        } for p in pos]
    except Exception as e:
        out["positions"] = []
        out["positions_error"] = f"{type(e).__name__}"

    # ---- today's orders at the broker ----------------------------------- #
    try:
        today = date.today().isoformat()
        orders = _alpaca_get(
            cfg, f"/v2/orders?status=all&limit=100&after={today}T00:00:00Z", paper)
        out["orders_today"] = [{
            "at": o.get("submitted_at", "")[:19],
            "symbol": o.get("symbol") or (o.get("legs") or [{}])[0].get("symbol", ""),
            "status": o.get("status"), "qty": o.get("qty"),
            "limit_price": o.get("limit_price"),
            "filled_avg_price": o.get("filled_avg_price"),
            "client_order_id": o.get("client_order_id", "")[:24],
        } for o in orders]
    except Exception as e:
        out["orders_today"] = []
        out["orders_error"] = f"{type(e).__name__}"

    # ---- what the harness itself did today ------------------------------ #
    # Distinct from broker orders: this includes REFUSALS, which never reach
    # Alpaca and are therefore invisible in the account. They are usually the
    # more interesting half -- "why did nothing trade today" lives here.
    try:
        recs = storage.JOURNAL.todays_submitted()
    except Exception:
        recs = []
    try:
        t = boto3.resource("dynamodb").Table(os.environ["STATE_TABLE"])
        day = date.today().isoformat()
        allrecs = t.query(KeyConditionExpression=boto3.dynamodb.conditions.Key(
            "pk").eq(f"ORDER#{day}")).get("Items", [])
    except Exception:
        allrecs = []
    out["activity"] = sorted([{
        "at": str(r.get("submitted_at", ""))[11:19],
        "token": str(r.get("token", "")),
        "status": str(r.get("status", "")),
        "reason": str(r.get("reason", ""))[:120],
        "order_id": str(r.get("order_id", ""))[:8],
    } for r in allrecs], key=lambda r: r["at"], reverse=True)[:25]
    out["counts"] = {
        "submitted_today": len(recs),
        "refused_today": sum(1 for r in out["activity"] if r["status"] == "refused"),
        "max_orders_per_day": limits.max_orders_per_day,
    }

    # ---- pending proposals (so the dashboard can act without typing) ----- #
    try:
        pend = storage.PENDING.all()
    except Exception:
        pend = {}
    now = datetime.now(timezone.utc)
    plist = []
    for tok, r in pend.items():
        try:
            age = (now - datetime.fromisoformat(r["created_at"])).total_seconds()
            left = max(0, int(limits.approval_ttl_seconds - age))
        except Exception:
            left = 0
        plist.append({
            "token": tok, "underlying": r.get("underlying"),
            "short": float(r.get("short_strike", 0)),
            "long": float(r.get("long_strike", 0)),
            "expiry": str(r.get("expiry", "")),
            "contracts": int(r.get("contracts", 0)),
            "credit": float(r.get("total_credit", 0)),
            "risk": float(r.get("total_risk", 0)),
            "expires_in_s": left,
        })
    out["pending"] = sorted(plist, key=lambda p: -p["expires_in_s"])

    # ---- risk envelope --------------------------------------------------- #
    # The daily budget the executor enforces is NET: deployed minus realised.
    # Reporting gross deployment here while the system gates on the net figure
    # would put a number on the dashboard that no control actually uses, which
    # is the same class of mistake as reporting default ceilings.
    risk_today = sum(float(r.get("total_risk", 0) or 0) for r in recs)
    realized_today = sum(float(r.get("realized_pnl", 0) or 0) for r in recs)
    consumed_today = risk_today - realized_today
    out["risk"] = {
        "budget_per_trade": limits.risk_budget(),
        "max_contracts": limits.max_contracts,
        "max_open_risk": limits.max_open_risk,
        "open_risk": sum(abs(p["cost_basis"]) for p in out.get("positions", [])),
        "deployed_today": round(risk_today, 2),
        "realized_today": round(realized_today, 2),
        # What the cap is actually compared against.
        "new_risk_today": round(consumed_today, 2),
        "max_new_risk_per_day": limits.max_new_risk_per_day,
    }

    return {
        "statusCode": 200,
        "headers": {
            "content-type": "application/json",
            # The dashboard is opened from a file:// URL, whose Origin is "null".
            # Echoing * is acceptable only because this endpoint is read-only and
            # already token-gated; it grants a reader nothing the token did not.
            "access-control-allow-origin": "*",
            "access-control-allow-headers": "authorization,content-type",
            "cache-control": "no-store",
        },
        "body": json.dumps(out, default=str),
    }
