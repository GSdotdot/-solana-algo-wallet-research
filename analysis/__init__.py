"""Module C: behaviour decoder (stage 2).

Reverse-engineers each tracked wallet's decision logic from a frozen export of the
collector's database. Bottom-up: per-wallet first, generic conclusions last.
The goal is DECODING into interpretable rules, not a black-box predictor.

Build order (each independently runnable, verified before the next):
  data         C0  load export -> clean per-event feature matrix
  features     --  the full causal feature inventory shared by every step
  profile      C1  per-wallet descriptive signature
  entry        C2  entry conditions (state at buy vs baseline)
  exit         C3  exit rules (time / price / signal driven), as a survival problem
  sizing       C4  position sizing
  cross_wallet C5  overlap + lead/follow + relative performance
  indicators   C6  indicator parameter sweep
  rules        C7  per-wallet rule extraction + cross-wallet synthesis

Every tuned value (model settings, sample minimums, feature windows, sweep grids,
verdict cut-offs) is read through common.params and is redacted in the public copy.
Wallets appear only under pseudonyms (WALLET_01, ...).
"""
