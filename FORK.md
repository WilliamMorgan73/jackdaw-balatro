# About this fork

Personal fork of [TylerFlar/jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro)
for a dissertation project on automating Balatro's shop phase with deep Q-learning.
`main` is upstream `main` plus engine fixes (one branch each, see [Fixes](#fixes)) and the
checking scripts below. Everything else is unchanged.

The RL code (shop environment, play policies, shop agents, training and the run logs the
dashboard reads) lives in a separate repo, `balatro-shop-rl`, which depends on this fork pinned
to a commit. Until fc213af it was here as `scripts/train_random_shop.py` and
`scripts/play_policy.py`.

## Additions

| File | What it does |
|---|---|
| `scripts/fuzz_engine.py` | Invariant fuzzer: full runs over every deck and stake (mixing `smart_agent`, random legal actions and mask sampling), checking every step for illegal or crashing actions, mask vs `get_legal_actions` disagreements, slot, card-ownership and price invariants, and that cash-out pays exactly the breakdown. `--easy` makes blinds trivial, so runs reach ante 8 and endless mode. Re-run it after merging upstream. |
| `scripts/verify_shop_rl.py` | Feasibility checks for a shop-only RL wrapper: raw-state keys, shop indexing, env `deepcopy` + RNG determinism, oracle timing, pack types, consumable targeting, joker slots. |

```bash
uv run python scripts/verify_shop_rl.py
uv run python scripts/fuzz_engine.py --games 300 && uv run python scripts/fuzz_engine.py --games 150 --easy --p-smart 0.2
```

## Keeping up with upstream

```bash
git fetch upstream
git switch main
git merge upstream/main      # then: git push
```

## Fixes

One fix per branch, branched from **upstream** `main` so it can be PR'd cleanly:

```bash
git fetch upstream
git switch -c fix/<name> upstream/main
# ...commit...
git push -u origin fix/<name>
gh pr create -R TylerFlar/jackdaw-balatro      # optional: offer it upstream
git switch main && git merge fix/<name>        # use it here straight away
```

Branches on this fork: `fix/rental-breakdown`, `fix/glass-destroy-rng`,
`fix/inflation-challenge`, `fix/cerulean-bell-forced-card`, `fix/pack-card-encoding`,
`fix/cerulean-bell-dedupe-actions` (built on the Cerulean branch) (merged into `main`); `fix/gym-reward-blind-beaten` and
`fix/use-consumables-in-pack` (superseded by upstream `ddc3d68`).

Fixes for the fork's issues #3–#5, one PR each to `main`, branched from upstream:
`fix/env-own-rng` (#5), `fix/aura-target` (#4, built on `fix/env-own-rng`) and
`fix/gym-obs-caps` (#3). The harness fixes for #1, #2, #6 and #7 moved to `balatro-shop-rl`
with the harness.

## Known issues and gotchas

- **Seeds depend on reset count per env object.** `BalatroEnvironment` seeds are
  `f"{prefix}_{n}"` where `n` counts `reset()` calls on that object; a fresh env per
  episode replays seed `_0`. Reuse one env or pass `reset(seed=...)`.
