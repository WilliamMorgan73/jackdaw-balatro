# About this fork

Personal fork of [TylerFlar/jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro)
for a dissertation project on automating Balatro's shop phase with deep Q-learning.
`main` is upstream `main` plus the additions below; everything else is unchanged.

## Additions

| File | What it does |
|---|---|
| `scripts/train_random_shop.py` | Shop-only training harness: a 17-action shop space, a pluggable play agent for every other phase (`--play random`, Jackdaw's `smart_agent`, or `--play strong` from `scripts/play_policy.py`), and a shop agent: `--shop random` (the baseline) `--shop qlearn` (tabular Q-learning, see [docs/q-learning-shop-agent.md](docs/q-learning-shop-agent.md)), or a fixed hand-written shop for the play-policy ceiling test: `--shop heuristic` (best affordable joker, planets, vouchers and packs only above `--interest-floor` $25) or `--shop smart` (`smart_agent`'s own shop logic, limited to the 17 actions). Use `--play smart` for shop experiments: `--play random` almost never beats the first blind, so it hardly ever reaches a shop. Writes `runs/<name>/` (`config.json`, `episodes.csv`, `steps.jsonl`, `rounds.jsonl`) for the balatro-run-analysis dashboard (separate repo). |
| `scripts/play_policy.py` | A stronger play policy (`--play strong`): `smart_agent` with exact `score_hand` scoring of the top candidate hands on a copy (boss debuffs and every joker count; fresh RNG, so no peeking), blind-aware play/discard and draw-aware discards (flush, straight, pairs). Packs, consumables and blind select are still `smart_agent`'s. |
| `scripts/fuzz_engine.py` | Invariant fuzzer: full runs over every deck and stake (mixing `smart_agent`, random legal actions and mask sampling), checking every step for illegal or crashing actions, mask vs `get_legal_actions` disagreements, slot, card-ownership and price invariants, and that cash-out pays exactly the breakdown. `--easy` makes blinds trivial, so runs reach ante 8 and endless mode. Re-run it after merging upstream. |
| `scripts/verify_shop_rl.py` | Feasibility checks for a shop-only RL wrapper: raw-state keys, shop indexing, env `deepcopy` + RNG determinism, oracle timing, pack types, consumable targeting, joker slots. `--play strong` times Phi with the stronger play policy. |
| `docs/q-learning-shop-agent.md` | How the tabular Q-learning shop agent works: state, actions, reward, update rule, pseudocode, diagrams, limitations. |
| `tests/scripts/test_q_shop_agent.py` | Unit tests for the Q-learning agent (legal actions only, update formula, epsilon schedule, reproducibility). |
| `tests/scripts/test_play_policy.py` | Unit tests for `scripts/play_policy.py`: draw odds, flush/straight targets, discard choice, exact scores matching the engine without touching the live state, the play-or-discard rule, legal actions only. |
| `tests/scripts/test_heuristic_shop_agents.py` | Unit tests for the ceiling-test shops (`--shop heuristic` and `--shop smart`): rule order, the interest floor, the 17-action mapping and fallback, legal actions only. |

```bash
uv run python scripts/train_random_shop.py --episodes 500 --play smart \
    --decks b_red,b_blue --stakes 1 --run-name my_run
uv run python scripts/train_random_shop.py --episodes 1000 --play smart --shop qlearn \
    --eval-every 100 --eval-episodes 20 --run-name qlearn_run
uv run python scripts/train_random_shop.py --episodes 1000 --play smart --shop heuristic \
    --eval-every 1000 --eval-episodes 500 --run-name heuristic_run
uv run python scripts/train_random_shop.py --episodes 1000 --play strong --shop heuristic \
    --eval-every 1000 --eval-episodes 500 --run-name strong_heuristic_run
uv run python scripts/verify_shop_rl.py [--play strong]
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

Fixes for the fork's issues #1–#7, one PR each to `main`. The env fixes are branched from
upstream: `fix/env-own-rng` (#5), `fix/aura-target` (#4, built on `fix/env-own-rng`) and
`fix/gym-obs-caps` (#3). The harness ones are fork-only: `fix/r-ante-max` (#1, #7; log_version 6,
built on `fix/env-own-rng`), `fix/sell-item-cost` (#6, built on `fix/r-ante-max`) and
`docs/random-play-warning` (#2).

## Known issues and gotchas

- **Seeds depend on reset count per env object.** `BalatroEnvironment` seeds are
  `f"{prefix}_{n}"` where `n` counts `reset()` calls on that object; a fresh env per
  episode replays seed `_0`. Reuse one env or pass `reset(seed=...)`.
