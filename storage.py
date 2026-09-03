"""Persistence seam between local runs and Lambda.

The trading logic in execution.py must behave identically whether it is running
from a launchd job against local files or in a container Lambda against
DynamoDB. Rather than fork that logic, everything it needs to persist goes
through the small interfaces here:

    Journal      the order/refusal audit log, plus the daily-cap queries
                 derived from it
    KillSwitch    the halt flag
    PendingStore  proposals awaiting your confirmation
    IVHistory     the daily ATM-IV series that bootstraps the iv_rank factor

The file implementations are the existing behaviour, unchanged, and remain the
default -- nothing about running locally changes. `aws/dynamo_store.py` supplies
the Lambda implementations.

One rule worth stating: the daily-cap and idempotency queries live BEHIND this
interface rather than being computed by the caller. If a caller had to remember
to check "have I already submitted this token", a caller would eventually
forget. The store answers the question.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
class Journal:
    def append(self, rec: dict) -> None:
        raise NotImplementedError

    def todays_submitted(self) -> List[dict]:
        """Orders with status 'submitted' stamped with today's date."""
        raise NotImplementedError

    def find_submitted(self, token: str) -> Optional[dict]:
        """The submitted order for this token, if one exists. Idempotency."""
        raise NotImplementedError

    def todays_realized_pnl(self) -> float:
        """Net realised P&L across closes stamped with today's date.

        Behind the interface for the same reason the daily-cap query is: a
        caller that had to remember to sum this would eventually forget, and
        the failure mode is a risk limit that silently measures the wrong
        number.
        """
        raise NotImplementedError


class KillSwitch:
    def engaged(self) -> bool:
        raise NotImplementedError


class PendingStore:
    def all(self) -> Dict[str, dict]:
        raise NotImplementedError

    def put(self, token: str, rec: dict) -> None:
        raise NotImplementedError

    def delete(self, token: str) -> None:
        raise NotImplementedError


class PositionStore:
    """Open positions the harness itself opened.

    Alpaca reports legs, not spreads, and it does not remember what credit you
    collected. Managing an exit needs the ENTRY credit -- profit target and stop
    are both expressed as multiples of it -- so the harness records its own
    positions at submit time rather than trying to reverse-engineer them from
    the broker later.
    """

    def all(self) -> Dict[str, dict]:
        raise NotImplementedError

    def put(self, key: str, rec: dict) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class IVHistory:
    def record(self, symbol: str, iv: float, on: Optional[date] = None) -> List[float]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# File implementations -- the existing local behaviour
# --------------------------------------------------------------------------- #
class FileJournal(Journal):
    def __init__(self, path="orders.jsonl"):
        self.path = Path(path)

    def _read(self) -> List[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue      # a truncated final line must not break the caps
        return out

    def append(self, rec: dict) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def todays_submitted(self) -> List[dict]:
        today = date.today().isoformat()
        return [r for r in self._read()
                if r.get("status") == "submitted"
                and str(r.get("submitted_at", "")).startswith(today)]

    def find_submitted(self, token: str) -> Optional[dict]:
        for r in self._read():
            if r.get("token") == token and r.get("status") == "submitted":
                return r
        return None

    def todays_realized_pnl(self) -> float:
        return sum(float(r.get("realized_pnl") or 0.0)
                   for r in self.todays_submitted())


class FileKillSwitch(KillSwitch):
    def __init__(self, path="KILL_SWITCH"):
        self.path = Path(path)

    def engaged(self) -> bool:
        return self.path.exists()

    def describe(self) -> str:
        return str(self.path.resolve())


class FilePendingStore(PendingStore):
    def __init__(self, path="pending.json"):
        self.path = Path(path)

    def all(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write(self, d: Dict[str, dict]) -> None:
        self.path.write_text(json.dumps(d, indent=2, default=str))

    def put(self, token: str, rec: dict) -> None:
        d = self.all()
        d[token] = rec
        self._write(d)

    def delete(self, token: str) -> None:
        d = self.all()
        d.pop(token, None)
        self._write(d)


class FilePositionStore(PositionStore):
    def __init__(self, path="positions.json"):
        self.path = Path(path)

    def all(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write(self, d):
        self.path.write_text(json.dumps(d, indent=2, default=str))

    def put(self, key: str, rec: dict) -> None:
        d = self.all(); d[key] = rec; self._write(d)

    def delete(self, key: str) -> None:
        d = self.all(); d.pop(key, None); self._write(d)


class FileIVHistory(IVHistory):
    def __init__(self, root=".cache/iv_history"):
        self.root = Path(root)

    def record(self, symbol: str, iv: float, on: Optional[date] = None) -> List[float]:
        self.root.mkdir(parents=True, exist_ok=True)
        on = on or date.today()
        p = self.root / f"{symbol.upper()}.json"
        hist = json.loads(p.read_text()) if p.exists() else {}
        hist[on.isoformat()] = iv
        p.write_text(json.dumps(hist, sort_keys=True))
        return [hist[k] for k in sorted(hist)]


# --------------------------------------------------------------------------- #
# Active bindings. Lambda swaps these at cold start; locally they are the files.
# --------------------------------------------------------------------------- #
JOURNAL: Journal = FileJournal()
KILL: KillSwitch = FileKillSwitch()
PENDING: PendingStore = FilePendingStore()
IV_HISTORY: IVHistory = FileIVHistory()
POSITIONS: PositionStore = FilePositionStore()


def use(journal: Journal = None, kill: KillSwitch = None,
        pending: PendingStore = None, iv_history: IVHistory = None,
        positions: PositionStore = None) -> None:
    """Rebind the active stores. Called once per Lambda cold start."""
    global JOURNAL, KILL, PENDING, IV_HISTORY, POSITIONS
    if journal is not None:
        JOURNAL = journal
    if kill is not None:
        KILL = kill
    if pending is not None:
        PENDING = pending
    if iv_history is not None:
        IV_HISTORY = iv_history
    if positions is not None:
        POSITIONS = positions
