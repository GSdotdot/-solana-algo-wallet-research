# Reverse-engineering algorithmic wallets on Solana microcaps

A research tool that tracked a small set of automated wallets trading Solana microcaps,
recorded the on-chain state of every token they touched, and tried to decode two things from
that record: **which token profiles attracted them**, and **how they entered and exited**.

This is a public showcase of a private project. The architecture and the methods are intact; the
watchlist, the data, the results and every tuned value are not here (see [§5](#5-note-on-redactions)).

## 1. What it is and why

While tracking microcaps I kept seeing the same pattern: one wallet with clearly automated
behaviour would enter a token, then another, then another, all converging on the same tokens
within a short window. If independent algorithms keep arriving at the same tokens, those
tokens probably carry a signal the algorithms are reading. That is a question with two sides:

* **The token side.** What did those tokens have in common at the moment each wallet bought?
* **The algorithm side.** What rule decides the entry, the size and the exit?

Answering either needs data that does not exist after the fact. A token's holder count, trader
mix and buy/sell flow at the second a wallet bought it cannot be rebuilt later, so the project
starts with a collector that records that state live, around every trade, and only then moves to
analysis.

The cross-wallet step of the analysis ([`analysis/cross_wallet.py`](analysis/cross_wallet.py))
tests the original observation directly: on tokens that more than one tracked wallet traded, who
entered first, how often, and by how long.

## 2. AI assistance

Code written with AI assistance; strategy design, filters and validation methodology are my own.

## 3. Pipeline and data schema

```
                    private watchlist (config/wallets.json, not in this repo)
                                        │
        ┌───────────────────────────────▼────────────────────────────────┐
 Stage 1│ collector/      one WebSocket per wallet  ──►  every transaction │
        │                 first touch of a token    ──►  one-time enrichment burst (REST)
        │                 while the token is live   ──►  tiered state polling (REST) + price
        │                 every trade               ──►  position tracking + ENTRY/EXIT labels
        └───────────────────────────────┬────────────────────────────────┘
                                        │ SQLite (WAL)  ──►  periodic CSV exports
        ┌───────────────────────────────▼────────────────────────────────┐
 Stage 2│ analysis/       C0 load ─ C1 profile ─ C2 entry ─ C3 exit ─ C4 sizing
        │                 C5 cross-wallet ─ C6 indicator sweep ─ C7 synthesis
        └───────────────────────────────┬────────────────────────────────┘
        ┌───────────────────────────────▼────────────────────────────────┐
 Stage 3│ store/          both collection windows merged into one de-duplicated
        │                 parquet store + a per-field reliability audit
        └────────────────────────────────────────────────────────────────┘
```

### Stage 1: collection (`collector/`)

The collector is driven by the wallets, not by a token scanner: a token only starts being
recorded when a tracked wallet first trades it.

| Step | Module | What it does |
|---|---|---|
| Ingest | `ws_client.py`, `wallet_txs.py` | One socket per wallet. Each transaction is parsed into BUY / SELL / token-to-token, with the quote currency kept separate so USDC amounts are never booked as SOL. |
| Discover | `subscriptions.py`, `rest_client.py` | On a wallet's first touch of a token: register it and fire a one-time enrichment burst (pre-discovery price history, recent swaps, top traders). |
| Track state | `stats_poller.py`, `stats_parser.py` | Poll each live token's holders, market cap, liquidity, trader-type mix and windowed buy/sell flow. Cadence depends on the token's lifecycle: densest while a wallet holds it, coarse once everyone has exited. An extra poll fires at every trade to capture the decision moment. |
| Price | `stats_poller.run_price`, `price_sockets.py`, `tick_archive.py` | A batch price call covers every live token. Tokens held by the fastest wallets get a dedicated socket streaming 1-second bars. |
| Archive | `snapshot_archive.py`, `features.py` | Every few seconds, each live token's merged state is written as one snapshot row, with live deltas. |
| Label | `positions.py`, `labeler.py` | Per-wallet position tracking labels each trade ENTRY, ADD, PARTIAL_EXIT, EXIT or FLIP (a quick re-entry), and computes realised PnL in USD net of priority fees. Each labelled trade is stored with the full token state at that instant. |
| Lifecycle | `subscriptions.py`, `config.pick_grace_tier` | After the last holder exits, a token is watched for a grace period sized by how actively it was traded, then dropped. |
| Budget | `credits.py`, `stats_poller.py` | Every REST call's API-credit cost is recorded. A pace guard throttles polling if spend runs ahead of a daily pace, and a hard ceiling stops REST polling entirely while the streams keep running. |
| Export | `export.py`, `excel_export.py`, `recompute_pnl.py` | Incremental CSV exports with a watermark, a summary workbook, and a replay tool that recomputes historical PnL through the same position engine after a fix. |

Data cost shaped this stage more than anything else. Cadence tiers, the at-trade poll, the
single dedicated price stream per fast-wallet token, and the two budget guards all exist to
spend credits where a wallet is actually making a decision.

### Stage 2: decoding (`analysis/`)

| Step | Module | Question |
|---|---|---|
| C0 | `data.py` | Load the frozen export into two tables: labelled **events** and reconstructed round-trip **positions**. |
| — | `features.py` | The shared feature factory: on-chain state at any moment, deltas over windows matched to each field's real refresh rate, the buy/sell flow ladder across timeframes, and per-token enrichment. Features use only data at or before each moment, with one documented exception: on a wallet's first trade of a token nothing was recorded beforehand, so the earliest state is the poll taken just after the buy (flagged, and bounded by a tolerance). |
| C1 | `profile.py` | Describe each wallet: hold-time regime, size consistency, win rate, and the features where its typical entry differs most from the group. |
| C2 | `entry.py` | Entry conditions. Compare the token's state when a wallet buys against random moments of the tracked tokens, and measure which feature families separate a buy from a non-buy. |
| C3 | `exit.py` | Exit rules as a survival problem: hold curves that count still-open positions, an "exit now vs keep holding" model at each moment in a trade, and a test of which family of signals best times the exit: time in trade, the price path since entry, or on-chain changes. |
| C4 | `sizing.py` | Position sizing: fixed or proportional, what it scales with, and whether bigger positions earn more. |
| C5 | `cross_wallet.py` | The convergence question: shared-token overlap, who enters first and by how much, results on co-traded tokens, and which untracked wallets keep appearing alongside. |
| C6 | `indicators.py` | A sweep of about fifty standard technical indicators across a ladder of periods, checking whether any setting clusters at entries or exits. |
| C7 | `rules.py` | Synthesis: one card per wallet, the cross-wallet ordering, and a table of which signal families carried out-of-sample predictive power. |

Validation follows one pattern. Train/test splits are grouped **by token**, so a token's rows
never sit on both sides. The headline scores (C2 entry, C3 within-hold timing, C4 sizing) are
reported next to a **shuffled-label null** that must come out at chance. C3 adds a temporal
holdout, and the indicator sweep selects on one set of tokens and confirms on another.

### Stage 3: unified store (`store/`)

Two collection windows were recorded in different formats (a frozen database and a series of CSV
exports that overlap by construction). `build_store.py` merges them into one de-duplicated
parquet store and reconciles row counts against each export's own summary.
`field_reliability.py` then audits every stored field: null rate, zero rate, and how often the
value actually changes.

### Data schema

Defined in [`collector/schema.py`](collector/schema.py). The per-window columns are generated
from one list so the tables and the parsers cannot drift apart.

| Table | One row per | Contents |
|---|---|---|
| `wallet_trades` | tracked-wallet transaction | direction, amounts, USD value, DEX, labels (entry / exit / add / partial / flip), realised PnL gross and net, hold time, trade sequence on the token, other tracked wallets holding at that moment |
| `token_snapshots` | live token, every few seconds | price, market cap, liquidity, holders, top-10 concentration, sniper / bundler / insider / dev counts and held share, bonding-curve progress and reserves, buy and sell counts and USD volume over eight windows (1m to 24h), live deltas |
| `labeled_events` | labelled trade | the trade plus the full token state at that instant, so the modelling matrix needs no joins |
| `token_identity` | discovered token | name, creation data, which wallet discovered it and when, and the three enrichment blobs (pre-discovery price history, recent swaps, top traders) |
| `price_ticks` | token, every second | 1-second OHLCV **bars** for tokens held by the fastest wallets. Despite the table name these are bars, not individual trades |
| `subscription_state` | tracked token | lifecycle bookkeeping: holders, grace tier, expiry |

[`sample_data/`](sample_data/) shows three of these tables in their real layout, filled with
**illustrative, made-up rows** (real rows proved re-identifiable even with sensitive columns
masked). Its README describes each column and what was hidden.

### Repository layout

```
collector/            stage 1: live collection
analysis/             stage 2: C0–C7 decoding
store/                stage 3: unified store + field audit
common/params.py      parameter loader (placeholders fail loudly if used)
config/               example watchlist and fee files (pseudonyms only), public DEX program IDs
config.example.yaml   every tunable, redacted, with a comment on what it controls
sample_data/          illustrative rows in the real table layouts + schema notes
tests/                offline tests on made-up values
```

```bash
pip install -r requirements.txt -r requirements-analysis.txt
pytest tests/          # offline; uses tests/test_config.yaml, whose values are invented
```

With only `config.example.yaml` present every module imports, but any use of a redacted value
raises an error naming the parameter. The collector and the analysis cannot run from this
repository: they need a private config, a watchlist and the data.

## 4. Status

**Paused.** Collection is the expensive part of this project, and the data costs outgrew what I
could sustain on a self-funded budget. The dataset is frozen at two collection windows.

What was learned so far, qualitatively:

* **The convergence is real and it has structure.** On tokens that several tracked wallets
  traded, entry order was far from random. A small number of wallets were consistently early
  and the others arrived later, a discover-then-pile-in ordering.
* **Selection was the most decodable part.** The token states the wallets bought could be told
  apart from other moments of the tracked tokens well above chance, out of sample, while the
  shuffled-label control stayed at chance.
* **Timing and exits were decodable for most wallets, not all.** One wallet showed no timeable
  entry at all, only selection. The very shortest holds sit below the resolution of the data.
* **Decodable is not the same as reproducible.** Knowing what the wallets select does not give
  you their tokens. Finding the same tokens independently means screening the whole live token
  universe at the same resolution, which is the data-cost problem again and the reason the
  project is paused rather than finished.
* **A first decode from one short window looked stronger than it was.** Rules derived from a
  single market regime had to be re-derived on the first window and checked once against the
  second, with the pass criteria fixed beforehand.
* **Signals that pick tokens do not necessarily time entries.** Several features separated
  bought tokens from the rest but were nearly constant inside a token, so they could not say
  when to buy. The two roles have to be measured separately.

And about the data itself:

* The provider's live payloads differed from its documentation, so parsers were built and tested
  against captured frames.
* On-chain fields refresh more slowly than they are sampled. Differencing them
  snapshot-to-snapshot produces mostly zeros, so the analysis rebuilds every delta over a window
  matched to that field's measured refresh rate.
* Looking only at closed round trips hides positions that never closed, so hold times are
  estimated with censoring.
* DEX fees and slippage are already inside the recorded trade amounts. Subtracting them again
  double-counts; only priority fees are deducted separately.
* In the second window the polled on-chain state stopped part-way through while prices kept
  streaming. That window is therefore split into a state lane and a price-only lane, and
  anything that needs state is validated on the first lane only.

## 5. Note on redactions

Tuned parameters, thresholds and data-provider details are redacted; placeholders mark where they apply.

Specific to this repository:

* **Wallets.** Every tracked wallet appears only as `WALLET_01…`, untracked ones as
  `EXT_WALLET_01…`, tokens as `TOKEN_A…`. No address, address fragment, transaction signature or
  token symbol is included. The watchlist itself is not included; only its loader and an
  example file with pseudonyms.
* **Results.** No output files, per-wallet findings or result figures are included. The
  hand-written conclusions inside the synthesis step are marked `[redacted: finding]`.
* **Sample data.** The sample rows are illustrative, not real: real rows could be traced back
  to their tokens even with times, prices and sizes masked.
* **Not included.** Private stages that replayed decoded rules in simulation and against
  held-out data, all deployment files, and all project notes.
* **Kept on purpose.** The wrapped-SOL and USDC mint addresses and the program IDs of well-known
  DEXes in `config/dex_programs.json` are public protocol constants.
