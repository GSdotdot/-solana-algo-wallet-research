"""Map the per-token stats REST responses into snapshot columns.

Division of labour between the three recurring / one-time calls:
  - token stats:   the workhorse. All 8 windows (buy/sell counts, USD totals,
    buy/sell split, change%) + holder + mcap/fdv/liquidity/supply + price, one call.
  - holder breakdown:   top-10 concentration + bundler/sniper/insider/dev (count + held%).
  - launchpad detail: graduated + bonding-curve progress + reserves.

The provider's JSON key for each neutral field name lives in config
(`data_provider.response_fields.*`, redacted in the public copy), so this module
only expresses WHAT is read and where it lands, not the provider's vocabulary.

All reads tolerate missing keys/None (each call can fail independently). Output keys
match token_snapshots column names, so the archive merges them directly.
build_stats() drops None so a failed/partial poll never clobbers a previously-good
cached value on merge.
"""

from __future__ import annotations

from typing import Any

from common.params import param

from .schema import WINDOWS

_TAGS = ("bundler", "sniper", "insider", "dev")

# Snapshot scalar columns read straight from the token stats.
_TOKEN_STATS_SCALARS = (
    ("holder", "int"), ("market_cap", "float"), ("fdv", "float"), ("liquidity", "float"),
    ("total_supply", "float"), ("circulating_supply", "float"), ("price", "float"),
)
# Per-window columns: (our column template, provider key template, type). Both
# templates are formatted with the window suffix {w}.
_TOKEN_STATS_WINDOWS = (
    ("buy_{w}", "buy_count", "int"), ("sell_{w}", "sell_count", "int"),
    ("volume_{w}_usd", "volume_usd", "float"),
    ("volume_buy_{w}_usd", "volume_buy_usd", "float"),
    ("volume_sell_{w}_usd", "volume_sell_usd", "float"),
    ("volume_{w}_change_percent", "volume_change_percent", "float"),
)


def _fields(group: str) -> dict[str, Any]:
    return param(f"data_provider.response_fields.{group}")


def _payload(resp: dict | None) -> dict:
    return (resp or {}).get(param("data_provider.response_fields.envelope")) or {}


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _to_int_bool(v: Any) -> int | None:
    return None if v is None else int(bool(v))


def from_token_stats(resp: dict | None) -> dict[str, Any]:
    """Token stats -> scalars + all 8 windows."""
    data = _payload(resp)
    if not data:
        return {}
    F = _fields("token_stats")
    cast = {"int": _i, "float": _f}
    out: dict[str, Any] = {col: cast[kind](data.get(F[col])) for col, kind in _TOKEN_STATS_SCALARS}
    for w in WINDOWS:
        for col_tpl, key, kind in _TOKEN_STATS_WINDOWS:
            out[col_tpl.format(w=w)] = cast[kind](data.get(F["windows"][key].format(w=w)))
    return out


def identity_from_token_stats(resp: dict | None) -> dict[str, Any]:
    """Token stats -> token_identity fields (name/symbol/decimals/logo/socials).

    The token-stats call is already polled per active token and carries the identity fields,
    so identity costs no extra call. None values are dropped so the caller's upsert
    never clobbers a previously-good identity column. NB: the token-stats call does NOT carry
    the bonding-curve/creation fields (creator, pool_address, source,
    creation_slot/tx, graduated_time, platform_id); those came from a token-screener
    stream that was retired, so they stop populating from that point."""
    data = _payload(resp)
    if not data:
        return {}
    F = _fields("identity")
    ext = data.get(F["extensions"]) or {}
    out: dict[str, Any] = {
        "token_name": data.get(F["token_name"]),
        "token_symbol": data.get(F["token_symbol"]),
        "decimals": _i(data.get(F["decimals"])),
        "logo_uri": data.get(F["logo_uri"]),
        "twitter": ext.get(F["twitter"]),
        "website": ext.get(F["website"]),
        "description": ext.get(F["description"]),
    }
    return {k: v for k, v in out.items() if v is not None}


def from_holder_breakdown(resp: dict | None) -> dict[str, Any]:
    """Holder breakdown -> top-10 concentration + trader-type tags."""
    data = _payload(resp)
    if not data:
        return {}
    F = _fields("holder_breakdown")
    top10 = (data.get(F["token_block"]) or {}).get(F["top10_block"]) or {}
    out: dict[str, Any] = {"top10_holder_percentage": _f(top10.get(F["percent_of_supply"]))}
    tags = {t.get(F["tag_name"]): t for t in (data.get(F["tags"]) or []) if isinstance(t, dict)}
    for tag in _TAGS:
        td = tags.get(tag) or {}
        out[f"{tag}_count"] = _i(td.get(F["holder_count"]))
        out[f"{tag}_held_percentage"] = _f(td.get(F["percent_of_supply"]))
    return out


def from_launchpad_detail(resp: dict | None) -> dict[str, Any]:
    """Launchpad detail -> bonding-curve state + reserves. Reserves are integer
    strings and are stored verbatim as TEXT (no float precision loss)."""
    F = _fields("launchpad_detail")
    mi = _payload(resp).get(F["info_block"]) or {}
    if not mi:
        return {}
    pool = mi.get(F["pool_block"]) or {}
    return {
        "graduated": _to_int_bool(mi.get(F["graduated"])),
        "progress_percent": _f(mi.get(F["progress_percent"])),
        "real_sol_reserves": pool.get(F["real_sol_reserves"]),
        "real_token_reserves": pool.get(F["real_token_reserves"]),
        "virtual_token_reserves": pool.get(F["virtual_token_reserves"]),
    }


def from_batch_price(resp: dict | None) -> dict[str, float]:
    """Batch price response -> {token_address: USD price}.

    The response maps each requested address to an entry holding the USD price (plus
    an update time we do not store today). Missing / null prices are dropped so a
    partial response never writes a bad price."""
    data = _payload(resp)
    key = _fields("batch_price")["price"]
    out: dict[str, float] = {}
    for addr, entry in data.items():
        if isinstance(entry, dict):
            v = _f(entry.get(key))
            if v is not None:
                out[addr] = v
    return out


def build_stats(token_stats: dict | None = None, holder: dict | None = None,
                launchpad: dict | None = None) -> dict[str, Any]:
    """Merge the three sources into snapshot columns, dropping None so a
    failed/partial poll never overwrites a previously-good cached value."""
    merged: dict[str, Any] = {}
    merged.update(from_token_stats(token_stats))
    merged.update(from_holder_breakdown(holder))
    merged.update(from_launchpad_detail(launchpad))
    return {k: v for k, v in merged.items() if v is not None}
