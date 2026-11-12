# Sample data (illustrative)

These files show **how the collector logged data**: the real table layouts, the real column
names in their real order, and which fields are populated on which kind of row.

**The rows are made up.** They were written by hand and generated, not extracted from the
database. An earlier version of this folder used real rows with sensitive columns masked. A
review showed that even the remaining columns (trade counts, the shape of a price path, the
order in which wallets traded a token) were enough to find the real tokens on-chain, and from
there the tracked wallets. So no real row is published.

How to read them:

* `XXXX` marks a field the collector **did** record but which is hidden here even in made-up
  form: times, sizes, prices, valuations and PnL. These are the fields that would let someone
  locate a trade.
* An empty cell means the collector stores nothing there on that kind of row (for example PnL
  on a buy), exactly as in the real tables.
* Every other value is invented but realistic in form. Pseudonyms (`WALLET_02`, `TOKEN_A`) are
  arbitrary and do not correspond to any real wallet or token.

## Files

### `transactions.csv`: tracked-wallet transactions (`wallet_trades` + the label from `labeled_events`)

30 rows over three tokens, in the shape the convergence pattern takes in the data: one tracked
wallet enters a token, then others follow, with exits, a partial exit, an add and a quick
re-entry (`FLIP`) along the way. The rows were written to illustrate that pattern.

This is the only true per-event table in the project: one row per on-chain trade.

| column | shown | meaning |
|---|---|---|
| `seq` | | order of the transaction on that token (1 = first) |
| `token`, `wallet` | pseudonym | arbitrary labels |
| `label` | | `ENTRY`, `ADD`, `PARTIAL_EXIT`, `EXIT`, `FLIP` (re-entry shortly after an exit) |
| `direction` | | `BUY` / `SELL` |
| `dex_program`, `tx_type` | | DEX the swap went through; transaction type |
| `has_pre_context` | | 1 if the token was already being recorded before this trade |
| `trade_sequence_on_token` | | running count of tracked-wallet trades on the token |
| `other_wallets_active` | | other tracked wallets holding the token at that moment |
| `is_entry` … `is_flip` | | the label as flags |
| `timestamp`, `slot_number` | `XXXX` | when the trade happened |
| `amount_token`, `amount_sol`, `price_per_token`, `volume_usd` | `XXXX` | trade size and price |
| `cumulative_position_tokens` | `XXXX` | position size after the trade |
| `pre_context_duration_sec`, `time_since_last_trade_any`, `time_since_last_trade_same_token` | `XXXX` | durations in seconds; the last is empty on a wallet's first trade on the token |
| `hold_duration_sec`, `position_pnl_usd`, `net_pnl_usd`, `position_pnl_sol` | `XXXX` / empty | realised on sells only, so empty on buys |
| `prio_bribe_fee_usd` | `XXXX` | the leg's priority fee, recorded on buys and sells |
| `tx_fee_sol`, `priority_fee_sol` | empty | never populated anywhere in the real data either |

### `token_features.csv`: token state snapshots (`token_snapshots`)

40 consecutive snapshots of one token, with every stored column except the two id columns that
contain the token address (90 columns). This is the token-features table the analysis reads.

| columns | shown | meaning |
|---|---|---|
| `seq`, `token` | | |
| `holder`, `top10_holder_percentage` | | holder count and top-10 concentration |
| `sniper_*`, `bundler_*`, `insider_*`, `dev_*` | | count and held share per trader type |
| `graduated` | | whether the token has left its bonding curve |
| `buy_<w>`, `sell_<w>` | | the token's trade counts (all traders) over each window, `1m` … `24h` |
| `volume_<w>_usd`, `volume_buy_<w>_usd`, `volume_sell_<w>_usd`, `volume_<w>_change_percent` | | the token's USD volume (all traders) over each window |
| `holder_delta_snap`, `holder_velocity_1m`, `*_pct_delta`, `top10_concentration_delta`, `buy_sell_ratio_*`, `volume_acceleration_1m` | | live deltas and ratios; the first row has no previous snapshot, so its deltas are empty |
| `timestamp`, `seconds_since_subscription` | `XXXX` | when the snapshot was taken |
| `price`, `market_cap`, `fdv`, `liquidity`, `total_supply`, `circulating_supply` | `XXXX` | price and valuation |
| `progress_percent`, `real_sol_reserves`, `real_token_reserves`, `virtual_token_reserves` | `XXXX` | bonding-curve position, which fixes the price |
| `liquidity_delta_snap`, `liquidity_velocity_1m`, `progress_velocity`, `sol_reserves_delta` | `XXXX` | deltas of hidden fields |
| `ohlcv_*` | empty | deprecated columns, no longer written |

The rows reproduce a property of the real table that shaped the analysis. The slow fields
refresh at the polling rate, not the snapshot rate, so they repeat across consecutive rows and
the live delta columns are zero on most rows. The analysis treats those columns as unusable and
rebuilds deltas over longer windows. (The refresh pattern in this file is irregular on purpose;
it does not encode the real polling cadence.)

### `price_bars_1s.csv`: 1-second price bars (`price_ticks`)

40 consecutive bars of a generated price path.

**These are 1-second bars, not ticks.** The table is called `price_ticks`, but each row is an
open/high/low/close bar for one second, not an individual trade. Bars were only stored for tokens
held by the fastest tracked wallets; every other token has the coarser snapshot price only.

| column | shown | meaning |
|---|---|---|
| `seq`, `token` | | |
| `open_index` … `close_index` | | prices expressed as an index (first open = 100) |
| `unix_time`, `received_at` | `XXXX` | bar time and arrival time |
| `volume`, `volume_usd` | `XXXX` | traded amount in the bar |

In the real table bars are stored as received and are not forward-filled, so a second with no
trade has no row.

## Tables not sampled

| table | why |
|---|---|
| `labeled_events` | it is `wallet_trades` joined to the token state at that instant; both halves are shown above |
| `token_identity` | names, symbols, creator and pool addresses, and raw provider blobs: identifying by nature |
| `subscription_state` | bookkeeping keyed by token address |
