"""SMS out, replies in. Twilio, polled -- no public endpoint required.

Two-way SMS normally means running a webhook Twilio can reach, which from a
laptop means a tunnel and an exposed port. Polling the inbound-message API gets
the same result with nothing listening, which is both simpler and a smaller
attack surface for a channel that can authorize trades.

THE REPLY GRAMMAR IS DELIBERATELY STRICT
----------------------------------------
Every proposal carries a short token, and a reply is only honoured if it
contains that token:

    Y A3F        confirm at the proposed size
    N A3F        decline
    A3F 2        confirm, but 2 contracts instead
    2 A3F        same

A bare "Y" does nothing. That matters: this inbox authorizes real orders, and
without token scoping any inbound message -- a wrong number, a carrier
notification, a delayed duplicate of an older reply -- could confirm whatever
happened to be pending. Tokens also make a reply unambiguous when two proposals
are outstanding at once.

An altered size is re-validated against the risk limits before submission. The
reply changes the requested quantity; it does not raise any ceiling.
"""
from __future__ import annotations

import json
import os
import random
import re
import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional


def new_token() -> str:
    """Short, unambiguous, phone-typable. No 0/O/1/I/L."""
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(3))


@dataclass
class Reply:
    token: str
    confirmed: bool
    contracts: Optional[int]      # None = use proposed size
    raw: str
    received_at: datetime


_TOKEN_RE = re.compile(r"\b([ABCDEFGHJKMNPQRSTUVWXYZ23456789]{3})\b", re.I)
_QTY_RE = re.compile(r"\b(\d{1,3})\b")


def parse_reply(body: str, known_tokens: List[str],
                received_at: Optional[datetime] = None) -> Optional[Reply]:
    """Parse an inbound SMS. Returns None if it does not clearly address a
    known pending proposal -- silence is the safe failure mode here."""
    if not body:
        return None
    text = body.strip().upper()
    known_upper = {t.upper(): t for t in known_tokens}

    token = None
    for cand in _TOKEN_RE.findall(text):
        if cand.upper() in known_upper:
            token = known_upper[cand.upper()]
            break
    if token is None:
        return None                      # no token => not consent to anything

    # Strip the token before hunting for a quantity, so a numeric-looking token
    # cannot be mistaken for a contract count.
    remainder = text.replace(token.upper(), " ")
    negative = re.search(r"\b(N|NO|CANCEL|STOP|DENY)\b", remainder)
    positive = re.search(r"\b(Y|YES|OK|CONFIRM)\b", remainder)
    qty_match = _QTY_RE.search(remainder)

    if negative:
        return Reply(token, False, None, body, received_at or datetime.now(timezone.utc))
    if qty_match:
        qty = int(qty_match.group(1))
        if qty <= 0:
            return Reply(token, False, None, body, received_at or datetime.now(timezone.utc))
        return Reply(token, True, qty, body, received_at or datetime.now(timezone.utc))
    if positive:
        return Reply(token, True, None, body, received_at or datetime.now(timezone.utc))
    return None                          # token but no clear intent


@dataclass
class Command:
    """A day-level instruction, as opposed to a reply about one trade."""
    kind: str                    # 'halt' | 'resume'
    raw: str


_HALT_RE = re.compile(r"\b(HALT|STOP\s*ALL|NO\s*TRADES|KILL)\b", re.I)
_RESUME_RE = re.compile(r"\b(RESUME|UNHALT|ALLOW)\b", re.I)


def parse_command(body: str, resume_token: Optional[str] = None) -> Optional[Command]:
    """Parse a day-level command. Returns None if the text is not one.

    ASYMMETRIC ON PURPOSE.

    HALT needs no token. Anyone who can write to the reply topic can stop your
    trading, and that is the correct trade-off: the worst case is a day of
    missed trades, and the best case is that in a genuine panic you can type
    one word from a locked phone without hunting for a code. Safety controls
    should be easy to pull.

    RESUME requires the token from the halt confirmation. Turning protection
    back ON is the direction where being wrong costs money, so it gets the
    friction. Same reasoning as the trade tokens: consent has to be specific.
    """
    if not body:
        return None
    text = body.strip()
    if _RESUME_RE.search(text):
        if resume_token and resume_token.upper() in text.upper():
            return Command("resume", body)
        return None                  # resume without the token is ignored
    if _HALT_RE.search(text):
        return Command("halt", body)
    return None


# --------------------------------------------------------------------------- #
class Notifier:
    def send(self, body: str, token: Optional[str] = None,
             halt_action: bool = False, alt_contracts: Optional[int] = None) -> None:
        """`token` lets channels that support interactive actions (ntfy) attach
        Confirm/Skip buttons. Channels without them ignore it."""
        raise NotImplementedError

    def poll_replies(self, known_tokens: List[str], since: datetime) -> List[Reply]:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """Prints instead of texting. Used for testing the whole loop with no
    Twilio account and no possibility of a message escaping."""

    def send(self, body: str, token: Optional[str] = None,
             halt_action: bool = False, alt_contracts: Optional[int] = None) -> None:
        print("\n--- SMS (console) " + "-" * 46)
        print(body)
        print("-" * 64)

    def poll_replies(self, known_tokens, since):
        return []


class TwilioNotifier(Notifier):
    """Real SMS. Credentials from env:

        TWILIO_ACCOUNT_SID
        TWILIO_AUTH_TOKEN
        TWILIO_FROM        the Twilio number, e.g. +15551234567
        ALERT_TO           your phone, e.g. +15559876543
    """

    def __init__(self):
        try:
            from twilio.rest import Client
        except ImportError:
            raise RuntimeError("pip install twilio")
        # Read from the environment, never hardcoded. This file is source and is
        # not covered by .gitignore, so a token pasted here rides along into the
        # first commit and stays in history after it is removed.
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        tok = os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_ = os.environ.get("TWILIO_FROM")
        self.to = os.environ.get("ALERT_TO")

        missing = [k for k, v in (("TWILIO_ACCOUNT_SID", sid), ("TWILIO_AUTH_TOKEN", tok),
                                  ("TWILIO_FROM", self.from_), ("ALERT_TO", self.to))
                   if not v]
        if missing:
            raise RuntimeError(f"missing Twilio env vars: {', '.join(missing)}")
        self.client = Client(sid, tok)

    def send(self, body: str, token: Optional[str] = None,
             halt_action: bool = False, alt_contracts: Optional[int] = None) -> None:
        # SMS segments at 160 chars; keep proposals inside one or two.
        self.client.messages.create(to=self.to, from_=self.from_, body=body[:900])

    def poll_replies(self, known_tokens: List[str], since: datetime) -> List[Reply]:
        """Inbound messages to the Twilio number since `since`.

        Filters on direction so an outbound alert can never be read back as a
        reply, and on sender so only your phone can authorize anything.
        """
        msgs = self.client.messages.list(to=self.from_,
                                         date_sent_after=since - timedelta(minutes=1),
                                         limit=50)
        out = []
        for m in msgs:
            if not str(getattr(m, "direction", "")).startswith("inbound"):
                continue
            if self.to and m.from_ != self.to:
                continue                 # only the registered handset
            r = parse_reply(m.body, known_tokens, getattr(m, "date_sent", None))
            if r:
                out.append(r)
        return out


class NtfyNotifier(Notifier):
    """Push notifications via ntfy.sh. Free, no account, no carrier registration.

    Alerts publish to one topic your phone subscribes to. The notification
    carries tappable action buttons that POST the reply back to a SECOND topic,
    which this class polls. Nothing on your machine has to be reachable, and
    there is no A2P registration to clear.

    SECURITY -- READ THIS BEFORE ARMING IT
    --------------------------------------
    A public ntfy.sh topic is readable and writable by anyone who knows its
    name. There is no sender authentication. Twilio at least verifies that a
    reply came from your handset; ntfy verifies nothing. What stands between a
    stranger and a confirmed order is:

        1. the reply topic name, which is why it must be long and random, and
        2. a live 3-character token that expires in 30 minutes.

    That is security by obscurity plus a short window. It is fine for alerts and
    fine for paper. For `--arm --live-money` it is materially weaker than SMS
    and you should either use Twilio, self-host ntfy with auth, or use a paid
    ntfy tier with access control. `alerts.py` warns when it sees this
    combination.

    Env:
        NTFY_TOPIC_ALERTS    topic your phone subscribes to
        NTFY_TOPIC_REPLIES   topic the action buttons publish to (keep secret)
        NTFY_SERVER          default https://ntfy.sh
    """

    def __init__(self):
        self.server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        self.alerts = os.environ.get("NTFY_TOPIC_ALERTS")
        self.replies = os.environ.get("NTFY_TOPIC_REPLIES")
        missing = [k for k, v in (("NTFY_TOPIC_ALERTS", self.alerts),
                                  ("NTFY_TOPIC_REPLIES", self.replies)) if not v]
        if missing:
            raise RuntimeError(f"missing env vars: {', '.join(missing)}. "
                               f"Generate topics with: python notify.py --new-topics")
        if len(self.replies) < 24:
            raise RuntimeError(
                "NTFY_TOPIC_REPLIES is too short to be unguessable. Anyone who "
                "guesses it can confirm your trades. Use --new-topics.")
        import requests            # noqa: F401  (fail early if absent)

    def send(self, body: str, token: Optional[str] = None,
             halt_action: bool = False, alt_contracts: Optional[int] = None) -> None:
        import requests
        headers = {"Title": "Trade proposal", "Priority": "high", "Tags": "chart"}
        if halt_action:
            headers["Title"] = "Market open - trading status"
            headers["Tags"] = "warning"
            headers["Actions"] = (
                f"http, HALT TODAY, {self.server}/{self.replies}, "
                f"method=POST, body='HALT', clear=true")
        elif token:
            # Tapping a button POSTs the reply straight to the reply topic, so
            # no endpoint of ours is ever exposed.
            #
            # ntfy allows three actions. Spending the third on an ALTERNATIVE
            # SIZE rather than something decorative: changing quantity is the
            # one response that otherwise requires typing free text, and typing
            # on a phone during market hours is exactly when you would fat-finger
            # it. `alt_contracts` is chosen by the caller -- normally 1 when the
            # proposal is larger, so the reflex action is always the smaller
            # position.
            #
            # SEPARATOR MATTERS: ntfy's short action format uses COMMAS between
            # the fields of one action and SEMICOLONS between actions. Joining
            # with commas makes it read the next action's leading "http" as a
            # field value and reject the whole message with
            # "actions invalid; term 'http' unknown".
            #
            # clear=true dismisses the notification the moment the button is
            # tapped. Without it a tap produces no visible change at all, and a
            # user with no feedback taps again -- observed in testing, 53 times.
            # Duplicate confirms are already harmless (the proposal leaves
            # pending on the first one, and the token is idempotent in
            # execution.py), but a control surface for real orders should never
            # leave you guessing whether your input registered.
            url = f"{self.server}/{self.replies}"
            acts = [f"http, Confirm, {url}, method=POST, body='Y {token}', clear=true"]
            if alt_contracts:
                acts.append(f"http, Confirm x{alt_contracts}, {url}, method=POST, "
                            f"body='{token} {alt_contracts}', clear=true")
            acts.append(f"http, Skip, {url}, method=POST, body='N {token}', clear=true")
            headers["Actions"] = "; ".join(acts)
        r = requests.post(f"{self.server}/{self.alerts}",
                          data=body.encode("utf-8"), headers=headers, timeout=20)
        # Never fail silently. A notifier that swallows errors reports success
        # while the alert never leaves the building -- which is exactly how a
        # malformed Actions header went unnoticed through three "passing" tests.
        if not r.ok:
            raise RuntimeError(f"ntfy publish failed: HTTP {r.status_code} {r.text[:200]}")

    def _poll_messages(self, since: datetime) -> List[tuple]:
        """Raw (body, when) pairs from the reply topic. Shared by trade replies
        and day-level commands so both see exactly the same message stream."""
        import requests
        r = requests.get(f"{self.server}/{self.replies}/json",
                         params={"poll": "1", "since": int(since.timestamp())},
                         timeout=25)
        if not r.ok:
            raise RuntimeError(f"ntfy poll failed: HTTP {r.status_code} {r.text[:200]}")
        out = []
        for line in r.text.splitlines():
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") != "message":
                continue
            out.append((msg.get("message", ""),
                        datetime.fromtimestamp(msg.get("time", 0), tz=timezone.utc)))
        return out

    def poll_commands(self, since: datetime,
                      resume_token: Optional[str] = None) -> List[tuple]:
        """-> [(Command, when)]. The timestamp is not decoration: the caller
        needs it to advance a watermark, without which every poll re-executes
        every command still inside the lookback window."""
        out = []
        for body, when in self._poll_messages(since):
            c = parse_command(body, resume_token)
            if c:
                out.append((c, when))
        return out

    def poll_replies(self, known_tokens: List[str], since: datetime) -> List[Reply]:
        import requests
        ts = int(since.timestamp())
        r = requests.get(f"{self.server}/{self.replies}/json",
                         params={"poll": "1", "since": ts}, timeout=25)
        # A failed poll must not look like "no replies". Without this, an outage
        # or a bad topic returns an error body, nothing parses as a message, and
        # the caller concludes you never answered -- your Confirm silently
        # disappears with no error anywhere.
        if not r.ok:
            raise RuntimeError(f"ntfy poll failed: HTTP {r.status_code} {r.text[:200]}")
        out = []
        for line in r.text.splitlines():
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") != "message":
                continue
            when = datetime.fromtimestamp(msg.get("time", 0), tz=timezone.utc)
            parsed = parse_reply(msg.get("message", ""), known_tokens, when)
            if parsed:
                out.append(parsed)
        return out


# --------------------------------------------------------------------------- #
def format_day_brief(state: dict, halt_token: str, reply_url: str = None) -> str:
    """The pre-open message: today's posture, and one tap to stop it.

    Leads with whether trading is enabled and where orders would go, because
    those are the two facts that decide whether you need to act. Everything
    else is context.
    """
    mode = "LIVE MONEY" if state.get("live") else "paper"
    enabled = "ARMED" if state.get("armed") else "alerts only"
    lines = [
        f"Market opens today. Status: {enabled} -> {mode}",
        f"account {state.get('account', '?')}",
        f"symbols {state.get('symbols', '?')}",
        f"risk/trade ${state.get('risk_budget', 0):.0f} | "
        f"max {state.get('max_contracts', '?')} contracts",
        f"open risk ${state.get('open_risk', 0):.0f} of "
        f"${state.get('max_open_risk', 0):.0f}",
    ]
    if state.get("halted"):
        lines.append("")
        lines.append(f"ALREADY HALTED for today. Reply 'RESUME {halt_token}' to undo.")
    else:
        lines.append("")
        lines.append("Tap HALT TODAY (or reply HALT) to stop all orders until "
                     "tomorrow. No token needed.")
    return "\n".join(lines)


def format_halt_confirmation(day, resume_token: str, by: str = "phone") -> str:
    return ("TRADING HALTED\n"
            f"No orders will be submitted for {day}.\n"
            f"Requested via {by}.\n"
            "The halt clears automatically overnight.\n"
            "--\n"
            f"To undo today, reply: RESUME {resume_token}")


def format_proposal(trade, score, limits) -> str:
    """The text itself. Everything needed to decide, and the reply grammar."""
    spread_frac = trade.half_spread_cost / trade.credit_per_contract
    return (
        f"[{trade.token}] {trade.underlying} put credit spread\n"
        f"{trade.short_strike:.0f}/{trade.long_strike:.0f}p exp {trade.expiry:%b %d}\n"
        f"SIZE {trade.contracts} contract(s)\n"
        f"credit ${trade.total_credit:.0f} | risk ${trade.total_risk:.0f}\n"
        f"short delta {abs(trade.short_delta):.2f} | spread {spread_frac:.0%} of credit\n"
        f"score {score.composite:+.2f} (coverage {score.coverage:.0%}, "
        f"{'mostly unvalidated' if score.coverage < 1 else ''})\n"
        f"sizing: {trade.sizing_note}\n"
        f"--\n"
        f"Reply Y {trade.token} to confirm, N {trade.token} to skip,\n"
        f"or '{trade.token} <n>' for n contracts.\n"
        f"Expires in {limits.approval_ttl_seconds // 60} min."
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if "--new-topics" in sys.argv:
        # 128 bits of entropy. The reply topic is the ONLY thing standing
        # between a stranger and a confirmed order on a public ntfy server, so
        # it is generated with secrets, not random, and it is long.
        alerts_t = "alpaca-alerts-" + secrets.token_urlsafe(9)
        replies_t = "alpaca-reply-" + secrets.token_urlsafe(16)
        print("Add to your env file, then subscribe your ntfy app to the ALERTS topic:\n")
        print(f'export NTFY_TOPIC_ALERTS="{alerts_t}"')
        print(f'export NTFY_TOPIC_REPLIES="{replies_t}"')
        print("\nSubscribe the phone app to the alerts topic only.")
        print("Keep the reply topic secret -- anyone who knows it can confirm a trade.")
    else:
        print("usage: python notify.py --new-topics")
