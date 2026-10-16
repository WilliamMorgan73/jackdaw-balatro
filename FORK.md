# About this fork

Personal fork of [TylerFlar/jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro)
for a dissertation project on automating Balatro's shop phase with deep Q-learning.
`main` is upstream `main` plus the additions below; everything else is unchanged.

## Additions

| File | What it does |
|---|---|
| `scripts/train_random_shop.py` | Shop-only training harness: a 17-action shop space, a pluggable play agent for every other phase (`--play random` or Jackdaw's `smart_agent`), random shop agent as the baseline. Writes `runs/<name>/` (`config.json`, `episodes.csv`, `steps.jsonl`, `rounds.jsonl`) for the balatro-run-analysis dashboard (separate repo). |
| `scripts/fuzz_engine.py` | Invariant fuzzer: full runs over every deck and stake (mixing `smart_agent`, random legal actions and mask sampling), checking every step for illegal or crashing actions, mask vs `get_legal_actions` disagreements, slot, card-ownership and price invariants, and that cash-out pays exactly the breakdown. `--easy` makes blinds trivial, so runs reach ante 8 and endless mode. Re-run it after merging upstream. |
| `scripts/verify_shop_rl.py` | Feasibility checks for a shop-only RL wrapper: raw-state keys, shop indexing, env `deepcopy` + RNG determinism, oracle timing, pack types, consumable targeting, joker slots. |

```bash
uv run python scripts/train_random_shop.py --episodes 500 --play smart \
    --decks b_red,b_blue --stakes 1 --run-name my_run
uv run python scripts/verify_shop_rl.py
uv run python scripts/fuzz_engine.py --games 300 && uv run python scripts/fuzz_engine.py --games 150 --easy --p-smart 0.2
```

### Evaluation on fixed seeds

`--eval-every N --eval-episodes K [--eval-seed-prefix EVAL]` plays K greedy
(`act(..., greedy=True)`, no `observe`) episodes before training and after every N
training episodes. The eval games are always `EVAL_0..K-1`, and game i gets deck
`decks[i % len(decks)]` and stake `stakes[(i // len(decks)) % len(stakes)]`. They are
written to `runs/<name>/eval/` (same files as a run, plus `eval_round` and
`train_episodes` columns; `config.json` lists the `eval_set`). Eval has its own env,
and its play agent's RNG is reseeded from each eval seed, so:

- turning eval on doesn't change the training episodes, and
- every run with the same `--decks/--stakes/--eval-*` plays the same eval games, so
  runs can be compared seed by seed.

`--group <name>` (saved in `config.json`) marks runs of one config with different
`--seed`s. The dashboard can average a group's runs and give CIs across seeds.

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
`fix/use-consumables-in-pack` (superseded by upstream `ddc3d68`);
`test/shop-rl-feasibility` (the history of the scripts above).

## Known issues and gotchas

- **Seeds depend on reset count per env object.** `BalatroEnvironment` seeds are
  `f"{prefix}_{n}"` where `n` counts `reset()` calls on that object; a fresh env per
  episode replays seed `_0`. Reuse one env or pass `reset(seed=...)`.
