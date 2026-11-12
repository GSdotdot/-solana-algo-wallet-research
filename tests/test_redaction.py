"""The public config must contain only placeholders, and must define every key the
test config defines (so nothing the code reads is missing from the template)."""

from pathlib import Path

import pytest
import yaml

from common.params import PLACEHOLDER, Redacted, RedactedParameterError, _wrap

ROOT = Path(__file__).resolve().parent.parent


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


def _keys(node, prefix=""):
    """Dotted paths of every mapping key (lists are leaves)."""
    out = set()
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.add(path)
            out |= _keys(v, path)
    return out


def _load(name):
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_example_config_holds_only_placeholders():
    leaks = [(k, v) for k, v in _leaves(_load("config.example.yaml")) if v != PLACEHOLDER]
    assert not leaks, f"non-placeholder values in config.example.yaml: {leaks}"


def test_example_config_covers_every_key_the_code_reads():
    example = _keys(_load("config.example.yaml"))
    test = _keys(_load("tests/test_config.yaml"))
    # a key that is a redacted leaf in the example may be a whole sub-tree in the test file
    missing = sorted(k for k in test
                     if k not in example and not any(k.startswith(e + ".") for e in example))
    assert not missing, f"keys missing from config.example.yaml: {missing}"


def test_redacted_value_fails_loudly_on_use_but_prints():
    r = _wrap(PLACEHOLDER, "section.key")
    assert isinstance(r, Redacted) and repr(r) == PLACEHOLDER and f"{r}" == PLACEHOLDER
    for use in (lambda: r + 1, lambda: r < 3, lambda: int(r), lambda: bool(r),
                lambda: list(r), lambda: r["k"]):
        with pytest.raises(RedactedParameterError, match="section.key"):
            use()


def test_wallet_examples_hold_only_pseudonyms():
    import json
    import re
    b58 = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,88}")
    for name in ("config/wallets.example.json", "config/wallet_fees.example.json"):
        text = (ROOT / name).read_text(encoding="utf-8")
        json.loads(text)
        assert not b58.search(text), f"address-like string in {name}"
