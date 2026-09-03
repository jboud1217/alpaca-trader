"""DynamoDB implementations of the storage interfaces.

Table layout -- one table, composite key, because these access patterns are
tiny and a single table avoids four sets of IAM grants and four things to
provision:

    pk                    sk                  purpose
    ------------------    ----------------    ---------------------------------
    PENDING               <token>             proposal awaiting confirmation
    ORDER#<yyyy-mm-dd>    <iso-ts>#<token>    submitted/refused audit record
    TOKEN                 <token>             idempotency marker for submissions
    IV#<SYMBOL>           <yyyy-mm-dd>        daily ATM IV observation
    CACHE                 <key>               refresh-lambda output (corp actions…)

Two details doing real work:

* PENDING items carry a DynamoDB TTL attribute, so proposal expiry is enforced
  by the database rather than by remembering to sweep. A confirmation arriving
  after the TTL finds nothing, which is the correct answer.

* ORDER records are partitioned by DAY. The daily-cap query is then a single
  partition read instead of a scan, and it stays that way after a year of
  history. The idempotency marker is a separate conditional write, because
  "have I already submitted this token" must be answerable atomically and not
  by scanning a day's worth of orders.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

import storage


def _clean(obj):
    """DynamoDB rejects float. Convert on the way in, restore on the way out."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


def _restore(obj):
    if isinstance(obj, Decimal):
        f = float(obj)
        return int(f) if f.is_integer() else f
    if isinstance(obj, dict):
        return {k: _restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v) for v in obj]
    return obj


class DynamoBackend:
    def __init__(self, table_name: str = None, client=None):
        self.table_name = table_name or os.environ["STATE_TABLE"]
        ddb = client or boto3.resource("dynamodb")
        self.t = ddb.Table(self.table_name)


# --------------------------------------------------------------------------- #
class DynamoJournal(storage.Journal, DynamoBackend):
    def append(self, rec: dict) -> None:
        token = rec.get("token", "unknown")
        stamp = rec.get("submitted_at") or datetime.now(timezone.utc).isoformat()
        day = str(stamp)[:10]
        item = _clean(dict(rec))
        item.update({"pk": f"ORDER#{day}", "sk": f"{stamp}#{token}"})
        self.t.put_item(Item=item)

        # Separate, conditional idempotency marker. Written only for real
        # submissions -- a refusal must not block a later legitimate retry.
        if rec.get("status") == "submitted":
            try:
                self.t.put_item(
                    Item=_clean({"pk": "TOKEN", "sk": token,
                                 "order_id": rec.get("order_id"),
                                 "submitted_at": stamp}),
                    ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)")
            except ClientError as e:
                if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                    raise
                # Someone beat us to it. find_submitted will now report it.

    def todays_submitted(self) -> List[dict]:
        day = date.today().isoformat()
        resp = self.t.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq(f"ORDER#{day}"))
        return [_restore(i) for i in resp.get("Items", [])
                if i.get("status") == "submitted"]

    def find_submitted(self, token: str) -> Optional[dict]:
        resp = self.t.get_item(Key={"pk": "TOKEN", "sk": token})
        item = resp.get("Item")
        return _restore(item) if item else None

    def todays_realized_pnl(self) -> float:
        # One partition read, same as the cap query -- ORDER records are
        # partitioned by day precisely so this stays a query and not a scan.
        return sum(float(r.get("realized_pnl") or 0.0)
                   for r in self.todays_submitted())


class SsmKillSwitch(storage.KillSwitch):
    """Kill switch as an SSM parameter, so halting is a one-line CLI call with
    no deploy and no code change:

        aws ssm put-parameter --name /alpaca/<env>/kill --value on --overwrite

    Fails CLOSED. If SSM cannot be read, the switch reports engaged -- an
    inability to verify that trading is permitted is not permission to trade.
    """

    def __init__(self, param: str = None, client=None):
        self.param = param or os.environ["KILL_PARAM"]
        self.ssm = client or boto3.client("ssm")
        self._cached = None
        self._at = 0.0

    def engaged(self) -> bool:
        if self._cached is not None and time.time() - self._at < 20:
            return self._cached
        try:
            v = self.ssm.get_parameter(Name=self.param)["Parameter"]["Value"]
            self._cached = str(v).strip().lower() in ("on", "true", "1", "yes")
        except Exception:
            self._cached = True          # fail closed
        self._at = time.time()
        return self._cached

    def describe(self) -> str:
        return f"SSM {self.param}"


class DayHalt(DynamoBackend):
    """A halt scoped to one trading day, set from your phone.

    Deliberately NOT the same thing as the global kill switch. The kill switch
    is a permanent stop you must remember to clear; a day halt expires on its
    own overnight. Those are different tools and conflating them means either
    you leave trading off for a week by accident, or you build the habit of
    clearing a "kill" every morning, which trains you to dismiss the control
    that matters most in a real incident.

    Stored with a DynamoDB TTL set to the end of the trading day in ET, so
    expiry needs no scheduled cleanup and no code that could fail to run.
    """

    KEY = "HALT"

    @staticmethod
    def _et_today() -> date:
        # ET without a tz database dependency: UTC-4 (EDT) is right for market
        # hours most of the year, and off by an hour in winter. A one-hour error
        # cannot change which *date* a halt applies to during RTH, which is all
        # this is used for.
        return (datetime.now(timezone.utc) - timedelta(hours=4)).date()

    @staticmethod
    def _end_of_day_epoch(d: date) -> int:
        # Midnight ET the following day, expressed as UTC epoch.
        midnight_et = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) \
            + timedelta(days=1, hours=4)
        return int(midnight_et.timestamp())

    def engaged(self, on: Optional[date] = None) -> bool:
        d = on or self._et_today()
        item = self.t.get_item(Key={"pk": self.KEY, "sk": d.isoformat()}).get("Item")
        if not item:
            return False
        # Filter on read as well as TTL: DynamoDB deletion lags by minutes and a
        # halt must not silently lapse inside that gap.
        return int(item.get("ttl", 0)) > int(time.time())

    def set(self, reason: str = "", by: str = "phone",
            on: Optional[date] = None) -> date:
        d = on or self._et_today()
        self.t.put_item(Item=_clean({
            "pk": self.KEY, "sk": d.isoformat(),
            "reason": reason, "by": by,
            "set_at": datetime.now(timezone.utc).isoformat(),
            "ttl": self._end_of_day_epoch(d)}))
        return d

    def clear(self, on: Optional[date] = None) -> date:
        d = on or self._et_today()
        self.t.delete_item(Key={"pk": self.KEY, "sk": d.isoformat()})
        return d


class CommandWatermark(DynamoBackend):
    """Timestamp of the newest command already acted on.

    Trade replies are self-consuming: acting on one deletes its PENDING row, so
    re-reading it later is harmless. Commands have no such row, and the reply
    topic is polled with a lookback window -- so without a watermark every poll
    re-executes every command still inside that window. Observed in production:
    a HALT and a RESUME both sitting in the 35-minute window made the responder
    halt, notify, resume, notify, once per minute, indefinitely.

    Storing "how far I have read" makes command processing idempotent the same
    way deleting the pending row does for confirmations.
    """

    KEY = "CMDWATERMARK"

    def get(self) -> Optional[datetime]:
        item = self.t.get_item(Key={"pk": self.KEY, "sk": "last"}).get("Item")
        if not item or not item.get("at"):
            return None
        try:
            return datetime.fromisoformat(item["at"])
        except ValueError:
            return None

    def set(self, when: datetime) -> None:
        # Monotonic: never move the watermark backwards, or a clock skew or an
        # out-of-order poll would replay commands we have already handled.
        cur = self.get()
        if cur and when <= cur:
            return
        self.t.put_item(Item={"pk": self.KEY, "sk": "last",
                              "at": when.isoformat()})


class CompositeKillSwitch(storage.KillSwitch):
    """Global SSM kill OR today's day-halt. One gate, checked in one place.

    execution.py asks a single question -- "am I allowed to trade" -- and gets a
    single answer. Adding a second independent check at the call sites would
    mean every future code path has to remember both, and one of them
    eventually will not.
    """

    def __init__(self, ssm_kill: "SsmKillSwitch", day_halt: DayHalt):
        self.ssm_kill = ssm_kill
        self.day_halt = day_halt
        self._why = ""

    def engaged(self) -> bool:
        if self.ssm_kill.engaged():
            self._why = f"SSM {self.ssm_kill.param}"
            return True
        try:
            if self.day_halt.engaged():
                self._why = f"day halt for {self.day_halt._et_today()}"
                return True
        except Exception:
            self._why = "day-halt table unreadable"
            return True                      # fail closed, same as the SSM path
        return False

    def describe(self) -> str:
        return self._why or "not engaged"


class DynamoPendingStore(storage.PendingStore, DynamoBackend):
    def __init__(self, table_name: str = None, client=None, ttl_seconds: int = 1800):
        DynamoBackend.__init__(self, table_name, client)
        self.ttl_seconds = ttl_seconds

    def all(self) -> Dict[str, dict]:
        resp = self.t.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("PENDING"))
        out = {}
        now = int(time.time())
        for item in resp.get("Items", []):
            # DynamoDB TTL deletion can lag by minutes. Filter on read so an
            # expired proposal is never actionable even if the row still exists.
            if int(item.get("ttl", 0)) and int(item["ttl"]) < now:
                continue
            rec = _restore({k: v for k, v in item.items()
                            if k not in ("pk", "sk", "ttl")})
            out[item["sk"]] = rec
        return out

    def put(self, token: str, rec: dict) -> None:
        item = _clean(dict(rec))
        item.update({"pk": "PENDING", "sk": token,
                     "ttl": int(time.time()) + self.ttl_seconds})
        self.t.put_item(Item=item)

    def delete(self, token: str) -> None:
        self.t.delete_item(Key={"pk": "PENDING", "sk": token})


class DynamoPositionStore(storage.PositionStore, DynamoBackend):
    """pk=POSITION, sk=<token>. No TTL: a position stays until it is closed."""

    def all(self) -> Dict[str, dict]:
        resp = self.t.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("POSITION"))
        return {i["sk"]: _restore({k: v for k, v in i.items() if k not in ("pk", "sk")})
                for i in resp.get("Items", [])}

    def put(self, key: str, rec: dict) -> None:
        item = _clean(dict(rec)); item.update({"pk": "POSITION", "sk": key})
        self.t.put_item(Item=item)

    def delete(self, key: str) -> None:
        self.t.delete_item(Key={"pk": "POSITION", "sk": key})


class DynamoIVHistory(storage.IVHistory, DynamoBackend):
    def record(self, symbol: str, iv: float, on: Optional[date] = None) -> List[float]:
        on = on or date.today()
        pk = f"IV#{symbol.upper()}"
        self.t.put_item(Item=_clean({"pk": pk, "sk": on.isoformat(), "iv": iv}))
        resp = self.t.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq(pk))
        rows = sorted(resp.get("Items", []), key=lambda i: i["sk"])
        return [float(r["iv"]) for r in rows]


class DynamoCache(DynamoBackend):
    """Slow-moving data written by the refresh lambda and read by scan."""

    def get(self, key: str) -> Optional[dict]:
        item = self.t.get_item(Key={"pk": "CACHE", "sk": key}).get("Item")
        return _restore(item.get("value")) if item else None

    def put(self, key: str, value: dict, ttl_seconds: int = 172_800) -> None:
        self.t.put_item(Item=_clean({
            "pk": "CACHE", "sk": key, "value": value,
            "ttl": int(time.time()) + ttl_seconds}))


# --------------------------------------------------------------------------- #
def install(table_name: str = None, kill_param: str = None,
            approval_ttl: int = 1800) -> DynamoCache:
    """Rebind storage.* to DynamoDB. Call once per cold start."""
    table_name = table_name or os.environ["STATE_TABLE"]
    halt = DayHalt(table_name)
    storage.use(
        journal=DynamoJournal(table_name),
        kill=CompositeKillSwitch(
            SsmKillSwitch(kill_param or os.environ["KILL_PARAM"]), halt),
        pending=DynamoPendingStore(table_name, ttl_seconds=approval_ttl),
        iv_history=DynamoIVHistory(table_name),
        positions=DynamoPositionStore(table_name),
    )
    return DynamoCache(table_name)


def day_halt(table_name: str = None) -> DayHalt:
    return DayHalt(table_name or os.environ["STATE_TABLE"])


def command_watermark(table_name: str = None) -> CommandWatermark:
    return CommandWatermark(table_name or os.environ["STATE_TABLE"])
