"""Lambda handlers: refresh (daily), scan (5-min), respond (1-min).

Split by CADENCE, not by data source. Corporate actions and earnings change
maybe quarterly; re-fetching them every five minutes for the same three symbols
would be pure waste and rate-limit pressure. Quotes and IV have to be fresh.
So:

    refresh   daily      corporate actions, earnings -> cache
    scan      5 min RTH  quotes, chain, news, score, size -> propose
    respond   1 min      poll replies, re-price, submit

Only `respond` is granted permission to trade. `scan` cannot place an order
even if its code were wrong, because its IAM role has no path to one and it
never constructs an Executor with armed=True. That separation is the point of
splitting them.

ARMED/LIVE come from SSM, not from the deployment, and are re-read on EVERY
invocation rather than cached per container -- see _boot(). Flipping either is
a CLI call that takes effect on the next invocation, which also means turning
trading off never requires a deploy to succeed first.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, "/var/task")          # repo modules live at the image root

import boto3

import dynamo_store
import events as ev
import execution as ex
import gamma as gm
import notify as nt
import storage
import weights as W

_PREFIX = os.environ.get("PARAM_PREFIX", "/alpaca/dev")
_cache = None
_ssm_client = None


def _ssm():
    """Lazily created. Building the client at import time turns any missing
    region or credential into an ImportError before a single line of handler
    code runs -- unhandleable, unloggable, and untestable outside a configured
    environment. Deferring it makes the same failure a normal exception."""
    global _ssm_client
    if _ssm_client is None:
        _ssm_client = boto3.client("ssm")
    return _ssm_client


# --------------------------------------------------------------------------- #
def _params(names, decrypt=True):
    """Fetch every parameter under the prefix in one paginated call.

    Not GetParameters: that caps at 10 names, and this config already has 11.
    Batching around the cap would work but leaves a landmine -- the twelfth key
    someone adds would break it again, at runtime, in the cloud. GetParametersByPath
    has no such limit and pages, so adding config never reintroduces this.
    """
    out = {}
    token = None
    while True:
        kw = {"Path": _PREFIX, "WithDecryption": decrypt, "MaxResults": 10}
        if token:
            kw["NextToken"] = token
        resp = _ssm().get_parameters_by_path(**kw)
        for p in resp.get("Parameters", []):
            out[p["Name"].rsplit("/", 1)[-1]] = p["Value"]
        token = resp.get("NextToken")
        if not token:
            break
    missing = [n for n in names if n not in out]
    if missing:
        raise RuntimeError(f"missing SSM parameters under {_PREFIX}: {missing}")
    return out


def _boot():
    """Load config and wire storage. Config is re-read EVERY invocation.

    It is tempting to cache this in a module global for the life of the warm
    container -- it is one SSM call and the credentials never change. That is a
    false economy, and it fails in the unsafe direction: `armed` and
    `live_money` live in the same parameter set, so a cached config means
    flipping armed=off does NOT stop a container that is already warm. You
    would issue the command, see it succeed in SSM, and the next invocation
    would trade anyway.

    A control you can turn on but not reliably turn off is not a control. One
    ~20ms SSM read per invocation, at one invocation per minute, is nothing
    against that.

    Only the storage BINDING is cached, because it is stateless wiring rather
    than a decision about whether to trade.
    """
    global _cache
    cfg = _params(["alpaca_key", "alpaca_secret", "ntfy_alerts", "ntfy_replies",
                   "armed", "live_money", "symbols", "equity", "risk_frac",
                   "max_contracts", "threshold"])
    os.environ["NTFY_TOPIC_ALERTS"] = cfg["ntfy_alerts"]
    os.environ["NTFY_TOPIC_REPLIES"] = cfg["ntfy_replies"]
    ttl = int(float(cfg.get("approval_ttl", os.environ.get("APPROVAL_TTL", 1800))))
    if _cache is None:
        _cache = dynamo_store.install(approval_ttl=ttl)
    else:
        # The store object is cached across invocations; its TTL must not be.
        storage.PENDING.ttl_seconds = ttl
    return cfg, _cache


def _limits(cfg) -> ex.RiskLimits:
    """Every cap comes from SSM. A strategy change that needs a redeploy is a
    strategy change you will not make, so none of these are literals."""
    def num(key, default, cast=float):
        try:
            return cast(cfg.get(key, default))
        except (TypeError, ValueError):
            return cast(default)
    return ex.RiskLimits(
        account_equity=num("equity", 25000),
        risk_per_trade_frac=num("risk_frac", 0.02),
        max_contracts=num("max_contracts", 2, int),
        max_risk_per_trade=num("max_risk_per_trade", 750),
        max_open_risk=num("max_open_risk", 2500),
        max_orders_per_day=num("max_orders_per_day", 3, int),
        max_new_risk_per_day=num("max_new_risk_per_day", 1500),
        # TTL and re-price tolerance are a pair. The TTL bounds how stale a
        # quote can be when you tap; the tolerance catches what slips through.
        # Shortening the TTL trades missed proposals for fewer refusals.
        approval_ttl_seconds=num("approval_ttl", os.environ.get("APPROVAL_TTL", 1800), int),
        reprice_tolerance_frac=num("reprice_tolerance", 0.10),
    )


def _flag(cfg, name) -> bool:
    return ex.flag_on(cfg, name)


def _clients(cfg, paper: bool):
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    k, s = cfg["alpaca_key"], cfg["alpaca_secret"]
    return {"option": OptionHistoricalDataClient(k, s),
            "stock": StockHistoricalDataClient(k, s),
            "news": NewsClient(k, s),
            "ca": CorporateActionsClient(k, s),
            "trading": TradingClient(k, s, paper=paper)}


def _ok(body):
    print(json.dumps(body, default=str))
    return {"statusCode": 200, "body": json.dumps(body, default=str)}


def guarded(name):
    """Push a notification when a handler throws.

    Until now a crash in `respond` was silent: CloudWatch recorded it, the alarm
    had no target, and the only symptom you would ever see is that tapping
    Confirm stopped doing anything. For the one component that decides whether
    orders happen, "fails quietly" is the worst possible failure mode.

    The alerts topic is read from the environment rather than SSM, because the
    most likely thing to be broken IS the SSM read -- a notifier that depends on
    the thing that just failed cannot report the failure.
    """
    def wrap(fn):
        def inner(event, context):
            try:
                return fn(event, context)
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                print(f"HANDLER FAILURE {name}: {detail}")
                topic = os.environ.get("NTFY_ALERTS_FALLBACK", "").strip()
                if topic:
                    try:
                        import requests
                        requests.post(
                            f"https://ntfy.sh/{topic}",
                            data=(f"LAMBDA FAILED: {name}\n{detail[:400]}\n"
                                  "Orders may not be processing. Check CloudWatch."
                                  ).encode(),
                            headers={"Title": f"alpaca-{name} error",
                                     "Priority": "urgent", "Tags": "rotating_light"},
                            timeout=10)
                    except Exception as ne:
                        print(f"could not notify about the failure: {ne}")
                raise      # still fail the invocation so metrics/alarms fire
        inner.__name__ = getattr(fn, "__name__", name)
        return inner
    return wrap


# --------------------------------------------------------------------------- #
@guarded("refresh")
def refresh(event, context):
    """Daily: cache the slow-moving inputs so scan does not refetch them."""
    cfg, cache = _boot()
    clients = _clients(cfg, paper=not _flag(cfg, "live_money"))
    symbols = [s.strip().upper() for s in cfg["symbols"].split(",") if s.strip()]

    out = {}
    for sym in symbols:
        rec = {}
        try:
            d, src = ev.next_ex_dividend(clients["ca"], sym)
            rec["next_ex_div"] = d.isoformat() if d else None
            rec["next_ex_div_source"] = src
        except Exception as e:
            rec["next_ex_div_error"] = f"{type(e).__name__}: {e}"
        cache.put(f"corpactions#{sym}", rec, ttl_seconds=172_800)
        out[sym] = rec

    # Dealer net gamma exposure, recorded daily.
    #
    # This is pure accumulation, not something scan reads. Alpaca serves CURRENT
    # open interest and no history, so the only way to ever test the conditional
    # signal from Baltussen et al. (JFE 2021) -- intraday momentum has beta=6.63
    # when dealers are net short gamma and beta=0.82, t=1.03, when they are long
    # -- is to start writing it down and wait. Every day this does not run is a
    # day that test can never cover.
    #
    # Failures are swallowed on purpose: a gamma snapshot is research data, and
    # it must never be able to break the refresh that scan actually depends on.
    gamma_out = {}
    for sym in symbols:
        try:
            snap = gm.compute(clients["trading"], clients["option"],
                              clients["stock"], sym, days_out=60)
            gm.record(snap)
            gamma_out[sym] = {"nge": round(snap.nge, 8),
                              "net_dollars": round(snap.net_gamma_dollars),
                              "dealers_short": snap.dealers_short_gamma,
                              "contracts": snap.contracts_used,
                              "oi_as_of": snap.oi_as_of}
        except Exception as e:
            gamma_out[sym] = {"error": f"{type(e).__name__}: {e}"}

    return _ok({"refreshed": out, "gamma": gamma_out})


# --------------------------------------------------------------------------- #
# The resume token is derived, not random: the respond lambda must be able to
# recompute it without shared state, and a halt confirmation sent at 08:45 has
# to still be answerable at 14:00 from a container that has since been recycled.
# It is not a secret against a determined attacker -- it is friction that makes
# "RESUME" specific to today rather than a word that re-enables trading whenever
# it happens to be typed.
def _resume_token(day) -> str:
    import hashlib
    h = hashlib.sha256(f"resume|{day}|{_PREFIX}".encode()).hexdigest().upper()
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(alphabet[int(h[i * 2:i * 2 + 2], 16) % len(alphabet)]
                   for i in range(3))


@guarded("daybrief")
def daybrief(event, context):
    """Pre-open: today's posture plus a one-tap halt."""
    cfg, cache = _boot()
    live = _flag(cfg, "live_money")
    armed = _flag(cfg, "armed")
    clients = _clients(cfg, paper=not live)
    limits = _limits(cfg)
    halt = dynamo_store.day_halt()

    try:
        acct = clients["trading"].get_account().account_number
    except Exception:
        acct = "unreachable"
    executor = ex.Executor(clients["trading"], clients["option"], limits,
                           armed=False, live=live)
    try:
        open_risk = executor.open_risk()
    except Exception:
        open_risk = -1.0

    day = halt._et_today()
    token = _resume_token(day)
    state = {
        "armed": armed, "live": live, "account": acct,
        "symbols": cfg.get("symbols"), "risk_budget": limits.risk_budget(),
        "max_contracts": limits.max_contracts, "open_risk": open_risk,
        "max_open_risk": limits.max_open_risk, "halted": halt.engaged(),
    }
    nt.NtfyNotifier().send(nt.format_day_brief(state, token), halt_action=True)
    return _ok({"brief": state, "day": str(day)})


def _process_commands(notifier, since, day, halt) -> list:
    """HALT / RESUME, handled before any trade confirmation.

    Ordering matters: if a halt and a confirmation arrive in the same polling
    window, the halt must win. Processing commands first means a stop you sent
    at 09:58 is in force before a 09:59 confirmation is considered, rather than
    racing it.
    """
    token = _resume_token(day)
    wm = dynamo_store.command_watermark()
    mark = wm.get()
    try:
        found = notifier.poll_commands(since, resume_token=token)
    except Exception as e:
        print(f"command poll failed: {type(e).__name__}: {e}")
        return []

    # Only commands NEWER than the watermark. Without this the lookback window
    # replays every command on every poll -- which in practice meant halting and
    # resuming once a minute, notifying each time.
    fresh = [(c, w) for c, w in found if mark is None or w > mark]
    acted = []
    if not fresh:
        return acted
    for c, when in sorted(fresh, key=lambda x: x[1]):
        if c.kind == "halt":
            if halt.engaged():
                acted.append({"cmd": "halt", "result": "already halted"})
                continue
            halt.set(reason=c.raw[:120], by="phone")
            notifier.send(nt.format_halt_confirmation(day, token))
            ex.record_refusal(f"HALT-{day}", f"day halt set from phone: {c.raw[:80]}")
            acted.append({"cmd": "halt", "result": f"halted for {day}"})
        elif c.kind == "resume":
            if not halt.engaged():
                acted.append({"cmd": "resume", "result": "not halted"})
                continue
            halt.clear()
            notifier.send(f"Trading RESUMED for {day}. Orders may be submitted again.")
            acted.append({"cmd": "resume", "result": f"resumed for {day}"})
        wm.set(when)     # advance per command, so a mid-batch crash cannot replay
    return acted


# --------------------------------------------------------------------------- #
@guarded("scan")
def scan(event, context):
    """Every 5 min during RTH: evaluate, size, publish a proposal."""
    cfg, cache = _boot()
    live = _flag(cfg, "live_money")
    clients = _clients(cfg, paper=not live)
    limits = _limits(cfg)

    if not clients["trading"].get_clock().is_open:
        return _ok({"skipped": "market closed"})

    notifier = nt.NtfyNotifier()
    # Deliberately unarmed. This lambda proposes; it must never be able to trade.
    executor = ex.Executor(clients["trading"], clients["option"], limits,
                           armed=False, live=live)

    symbols = [s.strip().upper() for s in cfg["symbols"].split(",") if s.strip()]
    threshold = float(cfg["threshold"])
    pending = storage.PENDING.all()
    results = {}

    # The backtest ran max_concurrent=1. Without this the live system would
    # stack spreads on the same underlying, which is a different and more
    # correlated risk profile than anything that was measured.
    open_pos = storage.POSITIONS.all()

    for sym in symbols:
        if any(r.get("underlying") == sym for r in pending.values()):
            results[sym] = "already pending"
            continue
        if any(r.get("underlying") == sym for r in open_pos.values()):
            results[sym] = "position already open"
            continue
        try:
            cal = _CachedEarnings(cache, sym)
            rec = _evaluate_cached(sym, clients, cal, cache, cfg)
        except Exception as e:
            results[sym] = f"error: {type(e).__name__}: {e}"
            continue
        if "error" in rec:
            results[sym] = rec["error"]
            continue

        score = rec["score"]
        if score.vetoed:
            results[sym] = f"vetoed: {'; '.join(score.vetoes)}"
            continue
        if score.composite < threshold:
            results[sym] = f"below threshold {score.composite:+.3f}"
            continue

        token = nt.new_token()
        while token in pending:
            token = nt.new_token()
        try:
            trade = ex.size_trade(token, sym, rec["chain"], rec["short_q"],
                                  rec["long_q"], limits,
                                  open_risk=executor.open_risk(),
                                  open_risk_this_underlying=executor.open_risk(sym))
        except ex.RiskRefusal as e:
            results[sym] = f"not sized: {e}"
            ex.record_refusal(token, str(e))
            continue

        # Offer the smaller size as the one-tap alternative when the proposal is
        # larger than one contract; otherwise offer stepping up to the cap.
        alt = 1 if trade.contracts > 1 else (
            limits.max_contracts if limits.max_contracts > 1 else None)
        notifier.send(nt.format_proposal(trade, score, limits),
                      token=token, alt_contracts=alt)
        item = {k: (v.isoformat() if isinstance(v, date) else v)
                for k, v in asdict(trade).items()}
        item["composite"] = score.composite
        storage.PENDING.put(token, item)
        pending[token] = item
        results[sym] = f"proposed {token}: {trade.summary()}"

    return _ok({"scan": results})


class _CachedEarnings:
    """Earnings from the cache table, falling back to UNKNOWN.

    UNKNOWN stays a veto in the cloud exactly as it is locally. A Lambda that
    cannot see an earnings calendar must not conclude there is no earnings.
    """

    def __init__(self, cache, symbol):
        self.cache = cache
        self.symbol = symbol

    def next_earnings(self, symbol, on=None):
        if symbol.upper() in ev.NO_EARNINGS_SYMBOLS:
            return None
        rec = self.cache.get(f"earnings#{symbol.upper()}")
        if not rec or not rec.get("date"):
            return ev.UNKNOWN
        d = date.fromisoformat(rec["date"])
        return d if d >= (on or date.today()) else None


def _evaluate_cached(symbol, clients, cal, cache, cfg=None):
    """Same scoring as alerts._evaluate, but reads corporate actions from the
    refresh cache instead of hitting the API on every five-minute tick."""
    stats = ev.underlying_stats(clients["stock"], symbol)
    if stats is None:
        return {"error": "no underlying history"}
    chain = ev.live_chain(clients["option"], symbol, stats.spot, max_dte=60)
    if chain is None:
        return {"error": "no live two-sided market"}

    target_dte = int((cfg or {}).get("target_dte", 45))
    wing_width = float((cfg or {}).get("wing_width", 5))
    short_delta = float((cfg or {}).get("short_delta", 0.30))
    expiry = chain.nearest_expiry(target_dte)
    if expiry is None:
        return {"error": "no expiry near 45 dte"}

    iv_atm = ev.atm_iv(chain, 30)
    slope = ev.term_structure_slope(chain)
    burst = ev.news_burst(clients["news"], symbol)

    ca = cache.get(f"corpactions#{symbol}") or {}
    exdiv = date.fromisoformat(ca["next_ex_div"]) if ca.get("next_ex_div") else None

    vetoes = []
    nxt = cal.next_earnings(symbol)
    if nxt == ev.UNKNOWN:
        vetoes.append("earnings UNKNOWN")
    elif isinstance(nxt, date) and nxt <= expiry:
        vetoes.append(f"earnings {nxt} before expiry")

    from strategies import PutCreditSpread
    pos = PutCreditSpread(short_delta=short_delta, wing_width=wing_width,
                          target_dte=target_dte).propose_entry(chain, 0)
    if pos is None:
        return {"error": "no strike pair matched"}
    sq = chain.find("put", max(l.strike for l in pos.legs), expiry)
    lq = chain.find("put", min(l.strike for l in pos.legs), expiry)
    if sq is None or lq is None:
        return {"error": "legs vanished from chain"}

    # Only short CALLS carry dividend early-assignment risk. A short put gains
    # from a dividend rather than being called away, so applying this to a put
    # credit spread would veto nearly every trade for a nonexistent risk.
    if (any(l.action == "sell" and l.right == "call" for l in pos.legs)
            and exdiv and exdiv <= expiry):
        vetoes.append(f"ex-div {exdiv} ({ca.get('next_ex_div_source','?')}) before expiry")

    credit = (sq.bid - lq.ask) * 100
    half = (0.5 * (sq.ask - sq.bid) + 0.5 * (lq.ask - lq.bid)) * 100
    if credit > 0:
        sv = W.spread_veto(half, credit)
        if sv:
            vetoes.append(sv)
    else:
        vetoes.append("no net credit")

    series = storage.IV_HISTORY.record(symbol, iv_atm) if iv_atm else []
    pct = None
    if len(series) >= 60 and iv_atm:
        pct = sum(1 for x in series if x <= iv_atm) / len(series)

    score = W.score({
        "iv_minus_rv":    W.norm_iv_minus_rv(iv_atm, stats.realized_vol_20d),
        "iv_rank":        W.norm_iv_rank(pct, len(series)),
        "term_slope":     W.norm_term_slope(slope),
        "trend_stress":   W.norm_trend_stress(stats.drawdown_20d, stats.pct_from_20d_ma),
        "spread_quality": W.norm_spread_quality(half, credit if credit > 0 else None),
        "news_intensity": W.norm_news_intensity(burst.intensity if burst else None),
    }, vetoes)
    return {"score": score, "chain": chain, "short_q": sq, "long_q": lq}


# --------------------------------------------------------------------------- #
@guarded("respond")
def respond(event, context):
    """Every minute: collect confirmations, re-price, submit.

    The only handler that can place an order, and only when SSM says armed.
    """
    cfg, cache = _boot()
    armed = _flag(cfg, "armed")
    live = _flag(cfg, "live_money")
    clients = _clients(cfg, paper=not live)
    limits = _limits(cfg)

    notifier = nt.NtfyNotifier()
    halt = dynamo_store.day_halt()
    since = datetime.now(timezone.utc) - timedelta(minutes=int(
        os.environ.get("REPLY_LOOKBACK_MIN", "35")))

    # Commands first: a HALT arriving in the same window as a confirmation must
    # take effect before that confirmation is considered.
    cmds = _process_commands(notifier, since, halt._et_today(), halt)

    pending = storage.PENDING.all()
    if not pending:
        return _ok({"pending": 0, "commands": cmds, "halted": halt.engaged(),
                    "auto_accept": ex.auto_accept_effective(cfg)})

    executor = ex.Executor(clients["trading"], clients["option"], limits,
                           armed=armed, live=live)
    # --- auto-accept ------------------------------------------------------
    # Treat every pending proposal as confirmed, without waiting for a reply.
    #
    # HARD INTERLOCK: this is IGNORED whenever live_money is on. The danger was
    # never auto-accept on paper -- it is auto-accept SURVIVING a later flip to
    # live, which would leave an unattended trader spending real money. Making
    # that impossible in code beats remembering to unset a flag, so the two
    # settings simply cannot both take effect.
    auto = ex.auto_accept_effective(cfg)
    if _flag(cfg, "auto_accept") and live:
        msg = ("auto_accept is ON and live_money is ON -- refusing to "
               "auto-confirm real-money trades. Proposals still require an "
               "explicit reply. Unset auto_accept to silence this.")
        print(f"[respond] REFUSED: {msg}")
        notifier.send(f"[!] {msg}")
        auto = False

    if auto:
        replies = [nt.Reply(token=t, confirmed=True, contracts=None,
                            raw="<auto_accept>",
                            received_at=datetime.now(timezone.utc))
                   for t in list(pending)]
        print(f"[respond] auto-accepting {len(replies)} proposal(s) (paper)")
    else:
        try:
            replies = notifier.poll_replies(list(pending), since)
        except Exception as e:
            return _ok({"error": f"poll failed: {type(e).__name__}: {e}"})

    acted = []
    for r in replies:
        rec = pending.get(r.token)
        if rec is None:
            continue                      # already handled, or expired
        pending.pop(r.token, None)

        if not r.confirmed:
            storage.PENDING.delete(r.token)
            ex.record_refusal(r.token, f"declined: {r.raw!r}")
            notifier.send(f"[{r.token}] skipped. Nothing submitted.")
            acted.append({"token": r.token, "result": "declined"})
            continue

        rec = {k: v for k, v in rec.items() if k != "composite"}
        rec["expiry"] = (date.fromisoformat(rec["expiry"])
                         if isinstance(rec["expiry"], str) else rec["expiry"])
        trade = ex.SizedTrade(**rec)

        if r.contracts is not None and r.contracts != trade.contracts:
            capped = min(r.contracts, limits.max_contracts)
            if capped * trade.max_loss_per_contract > limits.max_open_risk:
                capped = int(limits.max_open_risk // trade.max_loss_per_contract)
            if capped < 1:
                storage.PENDING.delete(r.token)
                notifier.send(f"[{r.token}] {r.contracts} exceeds limits. Not sent.")
                acted.append({"token": r.token, "result": "size refused"})
                continue
            if capped != r.contracts:
                notifier.send(f"[{r.token}] {r.contracts} exceeds caps, using {capped}.")
            trade.contracts = capped
            trade.total_credit = capped * trade.credit_per_contract
            trade.total_risk = capped * trade.max_loss_per_contract

        try:
            out = executor.submit(trade, dry_run=False)
            notifier.send(f"[{r.token}] SUBMITTED {trade.contracts}x @ "
                          f"${out['limit_price']:.2f} credit ({out['mode']}). "
                          f"Order {str(out['order_id'])[:8]}")
            acted.append({"token": r.token, "result": "submitted",
                          "order_id": out["order_id"]})
        except ex.RiskRefusal as e:
            notifier.send(f"[{r.token}] NOT submitted: {e}")
            ex.record_refusal(r.token, str(e))
            acted.append({"token": r.token, "result": f"refused: {e}"})
        except Exception as e:
            notifier.send(f"[{r.token}] submit ERROR: {type(e).__name__}")
            ex.record_refusal(r.token, f"{type(e).__name__}: {e}")
            acted.append({"token": r.token, "result": f"error: {type(e).__name__}"})
        storage.PENDING.delete(r.token)

    return _ok({"armed": armed, "live": live, "halted": halt.engaged(),
                "commands": cmds, "replies": len(replies), "acted": acted})


def _order_state(clients, order_id):
    """'filled' / 'dead' / 'live' for a submitted order, or 'live' if unreadable.

    Unreadable deliberately maps to 'live': if we cannot prove an order is
    finished, resubmitting could double the position, which is a worse failure
    than waiting one more cycle.
    """
    try:
        o = clients["trading"].get_order_by_id(order_id)
    except Exception:
        return "live"
    s = str(getattr(o, "status", "")).rsplit(".", 1)[-1].lower()
    if s == "filled":
        return "filled"
    if s in ("canceled", "cancelled", "expired", "rejected", "done_for_day"):
        return "dead"
    return "live"


def _settle_closing(positions, clients, notifier, acted):
    """Resolve positions that already have a close order in flight.

    A close is not a close until it FILLS. This used to delete the tracking
    record the instant submit_order returned, so a DAY limit that never filled
    left the spread open at the broker and invisible to this function forever
    -- nothing would ever try to exit it again, and it would run to expiry
    unmanaged. One IWM spread was orphaned exactly that way.

    Returns the tokens whose close is still working, so the caller skips them
    and never submits a second close for the same position.
    """
    in_flight = set()
    for token, p in list(positions.items()):
        c = p.get("closing")
        if not c:
            continue
        state = _order_state(clients, c.get("order_id"))
        if state == "filled":
            storage.POSITIONS.delete(token)
            positions.pop(token, None)
            notifier.send(f"[{token}] CLOSED — fill confirmed "
                          f"({c.get('reason', '')})")
            acted.append({"token": token,
                          "result": f"close confirmed: {c.get('reason', '')}"})
        elif state == "dead":
            # Died without filling. Drop the marker so the normal exit logic
            # re-evaluates at current quotes and submits a fresh order.
            p.pop("closing", None)
            storage.POSITIONS.put(token, p)
            notifier.send(f"[{token}] close did NOT fill "
                          f"({c.get('reason', '')}) — will retry at live quotes")
            acted.append({"token": token, "result": "close did not fill, retrying"})
        else:
            in_flight.add(token)
    return in_flight


def _reconcile(clients, positions, notifier, cache=None):
    """Alert on option positions the broker holds that the harness is not tracking.

    Nothing else compares broker state to POSITIONS. Without this an untracked
    spread is completely silent: absent from open_risk, never evaluated for an
    exit, and the first you hear of it is assignment.

    Notifies only when the untracked SET CHANGES. manage runs every 5 minutes
    for eight hours, so alerting unconditionally would send ~90 identical
    pushes a day and train you to swipe them away -- which would defeat the
    only alert that means "money is moving with nothing watching it".
    """
    try:
        held = clients["trading"].get_all_positions()
    except Exception as e:
        return {"reconcile_error": type(e).__name__}
    known = set()
    for p in positions.values():
        known.add(str(p.get("short_occ")))
        known.add(str(p.get("long_occ")))
    stray = sorted(
        str(h.symbol) for h in held
        if str(getattr(h, "asset_class", "")).endswith("option")
        and str(h.symbol) not in known)

    if stray:
        prev = None
        if cache is not None:
            try:
                prev = (cache.get("reconcile#untracked") or {}).get("symbols")
            except Exception:
                prev = None
        if prev != stray:
            notifier.send("UNTRACKED at broker: " + ", ".join(stray) +
                          "\nNot managed by this system. Close manually.")
        if cache is not None:
            try:
                cache.put("reconcile#untracked", {"symbols": stray}, ttl_seconds=86_400)
            except Exception:
                pass
    elif cache is not None:
        try:
            cache.put("reconcile#untracked", {"symbols": []}, ttl_seconds=86_400)
        except Exception:
            pass
    return {"untracked": stray}


# --------------------------------------------------------------------------- #
@guarded("manage")
def manage(event, context):
    """Exit management. The half of the strategy that was missing.

    `scan` opens positions and `respond` submits them, but until this existed
    nothing ever CLOSED anything. That is not a missing convenience -- it is a
    different strategy. The backtested edge lives in the exits: 32 of 44 closes
    were profit-target hits, and the friction sweep showed that removing the
    profit target turns +$596 into -$1,533. A system that only opens is "hold
    every spread to expiry with no stop", which is measurably worse than the
    thing that was tested.

    EXITS ARE NOT CONFIRMED BY YOU. Entries ask permission; exits do not.
    A stop-loss that waits for a tap is not a stop-loss -- the moment it matters
    most is exactly the moment you are in a meeting. The asymmetry is the same
    one HALT/RESUME uses: the risk-REDUCING direction gets less friction. Every
    exit still notifies, is journalled, and respects the kill switch and the
    day halt.
    """
    cfg, cache = _boot()
    live = _flag(cfg, "live_money")
    armed = _flag(cfg, "armed")
    clients = _clients(cfg, paper=not live)
    limits = _limits(cfg)

    positions = storage.POSITIONS.all()
    notifier = nt.NtfyNotifier()
    acted = []

    # Reconcile BEFORE the empty check. The orphan case is precisely the one
    # where POSITIONS is empty (or short an entry) while the broker still holds
    # the spread -- returning early on "no positions" is how it stayed hidden.
    recon = _reconcile(clients, positions, notifier, cache)

    if not positions:
        return _ok({"managed": 0, **recon})
    if not clients["trading"].get_clock().is_open:
        return _ok({"skipped": "market closed", "open": len(positions), **recon})

    in_flight = _settle_closing(positions, clients, notifier, acted)
    for token, p in list(positions.items()):
        if token in in_flight:
            continue
        try:
            verdict = _exit_verdict(p, clients, cfg, live)
        except Exception as e:
            acted.append({"token": token, "result": f"eval error: {type(e).__name__}"})
            continue
        if verdict is None:
            continue
        reason, close_credit = verdict

        if not armed:
            notifier.send(f"[{token}] EXIT SIGNAL ({reason}) but not armed — "
                          f"{p['underlying']} {p['short_strike']:.0f}/"
                          f"{p['long_strike']:.0f}p still open.")
            acted.append({"token": token, "result": f"signal only: {reason}"})
            continue
        try:
            out = _close_position(clients, p, reason, limits, live)
            # NOT a delete. Mark the close as in flight and keep the position
            # until the order is confirmed filled; _settle_closing removes it.
            p["closing"] = {"order_id": out["order_id"], "reason": reason,
                            "submitted_at": out["submitted_at"],
                            "limit_price": out["limit_price"]}
            storage.POSITIONS.put(token, p)
            notifier.send(f"[{token}] CLOSE SUBMITTED ({reason})\n"
                          f"{p['underlying']} {p['short_strike']:.0f}/"
                          f"{p['long_strike']:.0f}p x{p['contracts']}\n"
                          f"debit ${out['limit_price']*100:.0f}/contract vs "
                          f"${p['entry_credit_per_contract']:.0f} credit")
            acted.append({"token": token, "result": f"closed: {reason}",
                          "order_id": out["order_id"]})
        except ex.RiskRefusal as e:
            acted.append({"token": token, "result": f"exit refused: {e}"})
        except Exception as e:
            notifier.send(f"[{token}] EXIT FAILED ({reason}): {type(e).__name__}")
            acted.append({"token": token, "result": f"exit error: {type(e).__name__}"})
    return _ok({"open": len(positions), "acted": acted, **recon})


def _exit_verdict(p, clients, cfg, live):
    """-> (reason, cost_to_close_per_contract) or None. Mirrors _default_manage."""
    from alpaca.data.requests import OptionLatestQuoteRequest
    q = clients["option"].get_option_latest_quote(
        OptionLatestQuoteRequest(symbol_or_symbols=[p["short_occ"], p["long_occ"]]))
    s, l = q.get(p["short_occ"]), q.get(p["long_occ"])
    if s is None or l is None:
        return None                       # no market: cannot price an exit

    # Cost to flatten, crossing the spread (buy the short back at ask, sell the
    # long at bid). The pessimistic side, because that is what you would pay.
    cost = (float(s.ask_price) - float(l.bid_price)) * 100.0
    credit = float(p["entry_credit_per_contract"])
    pnl = credit - cost

    expiry = date.fromisoformat(str(p["expiry"])[:10])
    dte = (expiry - date.today()).days
    take = float(cfg.get("profit_take", 0.50))
    stop = float(cfg.get("stop_mult", 2.0))
    min_dte = int(cfg.get("min_dte", 21))

    if credit > 0 and pnl >= take * credit:
        return (f"profit target {int(take*100)}%", cost)
    if credit > 0 and pnl <= -stop * credit:
        return (f"stop {stop}x", cost)
    if dte <= min_dte:
        return (f"min_dte<={min_dte}", cost)
    return None


def _close_position(clients, p, reason, limits, live):
    """Reverse MLEG: buy_to_close the short, sell_to_close the long."""
    from alpaca.trading.enums import (OrderClass, OrderSide, PositionIntent,
                                      TimeInForce)
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.data.requests import OptionLatestQuoteRequest

    if storage.KILL.engaged():
        raise ex.RiskRefusal(f"kill switch engaged ({storage.KILL.describe()})")

    q = clients["option"].get_option_latest_quote(
        OptionLatestQuoteRequest(symbol_or_symbols=[p["short_occ"], p["long_occ"]]))
    s, l = q[p["short_occ"]], q[p["long_occ"]]
    debit = round((float(s.ask_price) - float(l.bid_price)), 2)
    if debit < 0:
        debit = 0.01          # a credit to close is fine; never send a negative limit

    req = LimitOrderRequest(
        qty=int(p["contracts"]), order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY, limit_price=debit,
        client_order_id=f"close-{p['token']}-{int(datetime.now(timezone.utc).timestamp())}",
        legs=[
            OptionLegRequest(symbol=p["short_occ"], ratio_qty=1,
                             side=OrderSide.BUY,
                             position_intent=PositionIntent.BUY_TO_CLOSE),
            OptionLegRequest(symbol=p["long_occ"], ratio_qty=1,
                             side=OrderSide.SELL,
                             position_intent=PositionIntent.SELL_TO_CLOSE),
        ])
    order = clients["trading"].submit_order(req)
    rec = {"token": f"CLOSE-{p['token']}", "status": "submitted",
           "submitted_at": datetime.now(timezone.utc).isoformat(),
           "mode": "live" if live else "paper", "reason": reason,
           "order_id": str(order.id), "limit_price": debit, "total_risk": 0.0}
    ex._journal(rec)
    return rec
