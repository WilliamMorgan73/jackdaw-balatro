# About this fork

Personal fork of [TylerFlar/jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro)
for a dissertation project on automating Balatro's shop phase with deep Q-learning.
`main` is upstream `main` plus the additions below; everything else is unchanged.

## Additions

| File | What it does |
|---|---|
| `scripts/train_random_shop.py` | Shop-only training harness: a 17-action shop space, a pluggable play agent for every other phase (`--play random` or Jackdaw's `smart_agent`), random shop agent as the baseline. Writes `runs/<name>/` (`config.json`, `episodes.csv`, `steps.jsonl`, `rounds.jsonl`) for the balatro-run-analysis dashboard (separate repo). |
| `scripts/verify_shop_rl.py` | Feasibility checks for a shop-only RL wrapper: raw-state keys, shop indexing, env `deepcopy` + RNG determinism, oracle timing, pack types, consumable targeting, joker slots. |

```bash
uv run python scripts/train_random_shop.py --episodes 500 --play smart \
    --decks b_red,b_blue --stakes 1 --run-name my_run
uv run python scripts/verify_shop_rl.py
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

Branches on this fork: `fix/gym-reward-blind-beaten`, `fix/use-consumables-in-pack`
(unmerged), `test/shop-rl-feasibility` (the history of the scripts above).

## Known issues and gotchas

- **`get_legal_actions()` offers pack picks the executor rejects.**
  `jackdaw/engine/actions.py` lists every `PickPackCard` (except full-slot jokers) without
  calling `pack_pick_block_reason()`, which the action mask and `_handle_pick_pack_card`
  both use. Agents built on engine legal actions (e.g. `smart_agent`) can pick The Fool
  with nothing to copy or Wheel of Fortune with no eligible joker and hit
  `IllegalActionError`. The factored `ActionMask` is correct. Both scripts work around it
  by filtering picks through the mask.
- **Seeds depend on reset count per env object.** `BalatroEnvironment` seeds are
  `f"{prefix}_{n}"` where `n` counts `reset()` calls on that object; a fresh env per
  episode replays seed `_0`. Reuse one env or pass `reset(seed=...)`.
