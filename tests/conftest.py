"""Test setup: point the parameter loader at tests/test_config.yaml (made-up values)
before any project module is imported, and make the repo root importable."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["WR_CONFIG"] = str(Path(__file__).resolve().parent / "test_config.yaml")
# Never let a developer's real environment leak into a test run.
for var in ("DATA_PROVIDER_API_KEY", "DATA_PROVIDER_REST_URL", "DATA_PROVIDER_WS_URL",
            "WRE_B_WALLETS", "WRE_MAX_WALLETS"):
    os.environ.pop(var, None)

from common import params  # noqa: E402

params.reload()
