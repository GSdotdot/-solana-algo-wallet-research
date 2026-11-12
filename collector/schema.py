"""SQLite schema: single source of truth for the collector's DDL.

No project dependencies (db.py imports and executes these). The per-window
volume/trade columns are *generated* from WINDOWS, so token_snapshots,
labeled_events and the stats parser can never silently drift apart. If a window
is added/removed, change it here only.

Schema decisions:
  - All 8 volume windows, full 6 fields each, on BOTH token_snapshots and
    labeled_events (labeled_events is the self-contained modelling matrix).
  - real/virtual reserves stored as TEXT (raw integer strings from the payload;
    avoids float precision loss. Module A parses them to int for deltas).
  - token_identity keeps the creation/bonding-curve fields that come for free
    with the provider's token payloads.
"""

# Rolling-window suffixes the provider reports for every token.
WINDOWS = ("1m", "5m", "30m", "1h", "2h", "4h", "8h", "24h")

# Applied on every connection open (db.py), not part of table DDL.
PRAGMAS = (
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
)


def _window_ddl(indent: str = "    ") -> str:
    lines: list[str] = []
    for w in WINDOWS:
        lines.append(f"{indent}buy_{w} INTEGER, sell_{w} INTEGER,")
        lines.append(f"{indent}volume_buy_{w}_usd REAL, volume_sell_{w}_usd REAL,")
        lines.append(f"{indent}volume_{w}_usd REAL, volume_{w}_change_percent REAL,")
    return "\n".join(lines)


# Shared token-state column blocks reused by token_snapshots and labeled_events.
_TOKEN_SCALAR_DDL = """\
    price REAL, market_cap REAL, fdv REAL, liquidity REAL,
    holder INTEGER, top10_holder_percentage REAL,
    total_supply REAL, circulating_supply REAL,
    sniper_count INTEGER, bundler_count INTEGER,
    insider_count INTEGER, dev_count INTEGER,
    sniper_held_percentage REAL, bundler_held_percentage REAL,
    insider_held_percentage REAL, dev_held_percentage REAL,
    graduated INTEGER, progress_percent REAL,
    real_sol_reserves TEXT, real_token_reserves TEXT,
    virtual_token_reserves TEXT,"""

_OHLCV_DDL = """\
    ohlcv_open REAL, ohlcv_high REAL, ohlcv_low REAL,
    ohlcv_close REAL, ohlcv_volume REAL,"""

# Module A derived features. Trailing comma omitted — these always come last.
_DERIVED_DDL = """\
    holder_delta_snap INTEGER, holder_velocity_1m REAL,
    liquidity_delta_snap REAL, liquidity_velocity_1m REAL,
    bundler_pct_delta REAL, sniper_pct_delta REAL,
    insider_pct_delta REAL, dev_pct_delta REAL,
    buy_sell_ratio_1m REAL, buy_sell_ratio_5m REAL,
    volume_acceleration_1m REAL, progress_velocity REAL,
    sol_reserves_delta REAL, top10_concentration_delta REAL"""


# ----------------------------------------------------------------------------
# Dataset 1 — wallet_trades
# ----------------------------------------------------------------------------
WALLET_TRADES_DDL = """\
CREATE TABLE IF NOT EXISTS wallet_trades (
    trade_id TEXT PRIMARY KEY,
    wallet_address TEXT NOT NULL,
    tx_signature TEXT NOT NULL UNIQUE,
    timestamp INTEGER NOT NULL,
    slot_number INTEGER,
    direction TEXT NOT NULL,            -- 'BUY','SELL','TOKEN_SWAP','OTHER'
    token_address TEXT,                 -- nullable: non-swap txs (transfers) have no single token
    token_symbol TEXT,
    amount_token REAL,
    amount_sol REAL,
    price_per_token REAL,
    pool_address TEXT,
    dex_program TEXT,
    tx_fee_sol REAL,
    priority_fee_sol REAL,
    volume_usd REAL,
    tx_type TEXT,                       -- 'swap','transfer','mint_add_liquidity',etc.
    has_pre_context INTEGER DEFAULT 0,  -- 0=false, 1=true
    pre_context_duration_sec INTEGER DEFAULT 0,
    snapshot_ref_id TEXT,
    raw_payload TEXT,                   -- full JSON
    -- Derived fields (populated post-ingestion, zero API cost):
    time_since_last_trade_any INTEGER,
    time_since_last_trade_same_token INTEGER,
    trade_sequence_on_token INTEGER,
    cumulative_position_tokens REAL,
    is_entry INTEGER, is_exit INTEGER,
    is_add INTEGER, is_partial_exit INTEGER, is_flip INTEGER,
    -- Realized PnL. USD is canonical (from volume_usd, currency-agnostic, already
    -- net-of-DEX). position_pnl_sol is precise SOL ONLY for fully-SOL-quoted
    -- positions, else NULL (it used to wrongly hold USDC amounts for USDC traders).
    position_pnl_usd REAL,              -- gross realized USD (net-of-DEX, pre prio+bribe)
    prio_bribe_fee_usd REAL,            -- this leg's priority+bribe (tip) cost, USD
    net_pnl_usd REAL,                   -- position_pnl_usd minus attributed prio+bribe
    position_pnl_sol REAL,              -- SOL-quoted positions only; NULL if USDC-touched
    hold_duration_sec INTEGER,
    other_wallets_active INTEGER
);"""

# ----------------------------------------------------------------------------
# Dataset 2 — token_snapshots (periodic archive, all 8 windows expanded)
# ----------------------------------------------------------------------------
TOKEN_SNAPSHOTS_DDL = f"""\
CREATE TABLE IF NOT EXISTS token_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    token_address TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    seconds_since_subscription INTEGER,
    -- Token scalar fields
{_TOKEN_SCALAR_DDL}
    -- Volume & trade windows (8 windows x 6 fields = 48 columns)
{_window_ddl()}
    -- OHLCV fields: DEPRECATED / NULL going forward. Per-token price lives in
    -- `price` (REST batch price, or the dedicated socket for targeted-B tokens)
{_OHLCV_DDL}
    -- Derived on-chain features (Module A)
{_DERIVED_DDL}
);"""

# ----------------------------------------------------------------------------
# Dataset 3 — token_identity (one-time per unique token)
# ----------------------------------------------------------------------------
TOKEN_IDENTITY_DDL = """\
CREATE TABLE IF NOT EXISTS token_identity (
    token_address TEXT PRIMARY KEY,
    token_name TEXT, token_symbol TEXT, logo_uri TEXT,
    decimals INTEGER,
    twitter TEXT, website TEXT, description TEXT,
    creation_time INTEGER, creator_wallet TEXT,
    platform_id TEXT,
    source TEXT,
    pool_address TEXT,
    graduated_time INTEGER,
    creation_slot INTEGER,
    creation_tx_hash TEXT,
    first_seen_timestamp INTEGER,
    discovered_via_wallet TEXT,
    ohlcv_backfill_json TEXT,   -- JSON array
    recent_trades_json TEXT,    -- JSON array
    top_traders_json TEXT,      -- JSON object
    exit_liquidity_json TEXT    -- JSON object
);"""

# ----------------------------------------------------------------------------
# Module B output — labeled_events (self-contained modelling matrix)
# ----------------------------------------------------------------------------
LABELED_EVENTS_DDL = f"""\
CREATE TABLE IF NOT EXISTS labeled_events (
    event_id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL REFERENCES wallet_trades(trade_id),
    snapshot_id TEXT REFERENCES token_snapshots(snapshot_id),
    wallet_address TEXT NOT NULL,
    token_address TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    label TEXT NOT NULL,                -- 'ENTRY','EXIT','ADD','PARTIAL_EXIT','FLIP'
    -- Snapshot context duplicated for self-contained analysis (exact state at trade time)
{_TOKEN_SCALAR_DDL}
    -- Volume & trade windows (full 8 windows, incl. buy/sell USD split)
{_window_ddl()}
    -- OHLCV
{_OHLCV_DDL}
    -- Derived features (Module A)
{_DERIVED_DDL},
    -- Trade metadata
    direction TEXT, amount_sol REAL, amount_token REAL,
    price_per_token REAL, volume_usd REAL,
    has_pre_context INTEGER, pre_context_duration_sec INTEGER
);"""

# ----------------------------------------------------------------------------
# Internal bookkeeping — subscription_state
# ----------------------------------------------------------------------------
SUBSCRIPTION_STATE_DDL = """\
CREATE TABLE IF NOT EXISTS subscription_state (
    token_address TEXT PRIMARY KEY,
    subscribed_at INTEGER NOT NULL,
    screener_active INTEGER DEFAULT 1,
    price_active INTEGER DEFAULT 1,
    last_wallet_trade_at INTEGER,
    wallets_holding TEXT,               -- JSON array of wallet addresses
    grace_tier TEXT,                    -- 'short','medium','long', or NULL
    grace_expires_at INTEGER            -- NULL if not in grace
);"""

# ----------------------------------------------------------------------------
# Raw 1-second price-bar archive, written live by the targeted-B per-token price
# sockets (collector.tick_archive), rolling-window pruned. The provider's price
# channel is single-active per connection, so only the targeted-B scalpers' held
# tokens get 1s bars; every other token's price comes from the REST batch price poll.
# Despite the table name, each row is a 1s OHLCV bar, not an individual trade.
# ----------------------------------------------------------------------------
PRICE_TICKS_DDL = """\
CREATE TABLE IF NOT EXISTS price_ticks (
    token_address TEXT NOT NULL,
    unix_time INTEGER NOT NULL,        -- 1s bar time reported by the provider
    ohlcv_open REAL, ohlcv_high REAL, ohlcv_low REAL, ohlcv_close REAL,
    volume REAL, volume_usd REAL,
    received_at INTEGER,
    PRIMARY KEY (token_address, unix_time)   -- OR REPLACE keeps the final bar value
);"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_wt_wallet_ts ON wallet_trades(wallet_address, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_wt_token_ts  ON wallet_trades(token_address, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_wt_direction ON wallet_trades(direction, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ts_token_ts  ON token_snapshots(token_address, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ts_ts        ON token_snapshots(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_le_wallet    ON labeled_events(wallet_address, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_le_token     ON labeled_events(token_address, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_le_label     ON labeled_events(label);",
    "CREATE INDEX IF NOT EXISTS idx_pt_time      ON price_ticks(unix_time);",  # for rolling-window prune
)

# Additive column migrations for pre-existing DBs: CREATE TABLE IF NOT EXISTS does
# NOT add columns to a table that already exists, so a deployed DB would otherwise
# silently drop these on insert. Each = (table, column, type); applied only if the
# column is missing (idempotent). New nullable columns only — never a rewrite.
MIGRATIONS = (
    ("wallet_trades", "position_pnl_usd", "REAL"),
    ("wallet_trades", "prio_bribe_fee_usd", "REAL"),
    ("wallet_trades", "net_pnl_usd", "REAL"),
)

# Ordered so FK-referenced tables (wallet_trades, token_snapshots) exist first.
TABLES = (
    ("wallet_trades", WALLET_TRADES_DDL),
    ("token_snapshots", TOKEN_SNAPSHOTS_DDL),
    ("token_identity", TOKEN_IDENTITY_DDL),
    ("labeled_events", LABELED_EVENTS_DDL),
    ("subscription_state", SUBSCRIPTION_STATE_DDL),
    ("price_ticks", PRICE_TICKS_DDL),
)


def all_statements() -> list[str]:
    """Ordered DDL: tables first (FK order), then indexes. PRAGMAs applied separately."""
    return [ddl for _, ddl in TABLES] + list(INDEXES)


if __name__ == "__main__":
    # `python -m collector.schema` dumps the resolved DDL for review.
    for stmt in all_statements():
        print(stmt + "\n")
