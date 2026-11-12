"""Entrypoint: `python -m collector`.

Requires the data-provider variables in .env (see .env.example), a private
config.local.yaml with real values, and config/wallets.json populated. This opens
real provider subscriptions and spends API credits: do not run at scale without
intent. For a bounded local smoke run, set WRE_MAX_WALLETS=1 and
WRE_RUN_SECONDS=<n>.
"""

from __future__ import annotations

import asyncio

from .orchestrator import Orchestrator
from .util import setup_logging


def main() -> None:
    setup_logging()
    orch = Orchestrator()
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
