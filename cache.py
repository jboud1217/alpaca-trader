"""Disk cache + rate limiting for the Alpaca data path.

Reconstructing an as-of option chain is API-call expensive: a two-year backtest
touches hundreds of trading days and thousands of contracts. Without a cache
every parameter sweep re-downloads the same history and you burn the rate limit
on data you already have. With one, the first run is slow and every run after
it is local.

The cache is keyed by content, not by time, and option history for a past date
does not change -- so entries never expire. Delete the directory to refetch.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional


class DiskCache:
    def __init__(self, root: str = ".cache/alpaca"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        try:
            with p.open() as fh:
                return json.load(fh)["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None    # corrupt entry: treat as a miss and let it refetch

    def put(self, namespace: str, key: str, value: Any) -> None:
        p = self._path(namespace, key)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w") as fh:
            json.dump({"key": key, "value": value}, fh)
        os.replace(tmp, p)   # atomic: a killed run never leaves a half-written entry

    def get_or_fetch(self, namespace: str, key: str, fetch: Callable[[], Any]) -> Any:
        hit = self.get(namespace, key)
        if hit is not None:
            return hit
        value = fetch()
        self.put(namespace, key, value)
        return value

    def stats(self) -> dict:
        out = {}
        for ns in sorted(p for p in self.root.iterdir() if p.is_dir()):
            files = list(ns.glob("*.json"))
            out[ns.name] = {
                "entries": len(files),
                "mb": round(sum(f.stat().st_size for f in files) / 1e6, 2),
            }
        return out


class RateLimiter:
    """Simple sliding-window throttle.

    Alpaca's free data tier allows 200 requests/minute. Default is set below
    that because bursting to the ceiling gets you 429s that cost more time than
    the throttle does.
    """

    def __init__(self, per_minute: int = 180):
        self.per_minute = per_minute
        self._times: list = []

    def wait(self) -> None:
        now = time.monotonic()
        self._times = [t for t in self._times if now - t < 60.0]
        if len(self._times) >= self.per_minute:
            sleep_for = 60.0 - (now - self._times[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self._times = [t for t in self._times if now - t < 60.0]
        self._times.append(now)
