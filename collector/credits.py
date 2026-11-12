"""API-credit accounting (this code spends money on every REST call).

Tracks credits per REST call type and a running total, so the pace guard and hard
ceiling in stats_poller can act on it and graceful shutdown can log exactly what
the run cost. WebSocket streaming is billed differently and is not known
per-message client-side; REST credits are tracked precisely and WS is treated as
estimate-only.

"""

from __future__ import annotations


class CreditTracker:
    def __init__(self) -> None:
        self.total: float = 0.0
        self.by_endpoint: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def add(self, endpoint: str, credits: float) -> None:
        self.total += credits
        self.by_endpoint[endpoint] = self.by_endpoint.get(endpoint, 0.0) + credits
        self.calls[endpoint] = self.calls.get(endpoint, 0) + 1

    def summary(self) -> dict:
        return {
            "total_credits": round(self.total, 2),
            "by_endpoint": {k: round(v, 2) for k, v in sorted(self.by_endpoint.items())},
            "calls": dict(sorted(self.calls.items())),
        }
