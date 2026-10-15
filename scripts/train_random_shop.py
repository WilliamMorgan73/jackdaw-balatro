"""Shop-only training harness with a random play agent and a random shop agent.

The shop agent acts in a 17-action space (5 buy + 6 sell joker + 2 sell
consumable + 2 use consumable + reroll + skip).  Every other phase (blind
select, play/discard, cash out, pack opening) is handled by the play agent.

Random agents don't learn: this is the training loop's plumbing plus the
random baseline.  A learning shop agent (e.g. a Dueling DQN) implements the
same ``act`` / ``observe`` interface and drops in for ``RandomShopAgent``.

Each run writes ``runs/<run-name>/``: ``config.json``, ``episodes.csv`` (one
row per episode), ``steps.jsonl`` (one line per shop decision, with the actions
that were legal), ``rounds.jsonl`` (one line per blind, with the cash-out
breakdown) and ``shops.jsonl`` (what each shop offered, on entry and after
rerolls).  Visualise them with balatro-run-analysis.

Usage::

    uv run python scripts/train_random_shop.py --episodes 200
    uv run python scripts/train_random_shop.py --episodes 200 --play smart
    uv run python scripts/train_random_shop.py --play smart \\
        --decks b_red,b_blue,b_black --stakes 1,2 --run-name smart_mixed
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from jackdaw.engine.actions import GamePhase
from jackdaw.env import BalatroEnvironment, DirectAdapter
from jackdaw.env.action_space import ActionMask, FactoredAction, engine_action_to_factored
from jackdaw.env.action_space import ActionType as AT

# ---------------------------------------------------------------------------
# 17-action shop space
# ---------------------------------------------------------------------------

N_ACTIONS = 17
BUY_CARD = (0, 1)
BUY_PACK = (2, 3)
BUY_VOUCHER = 4
SELL_JOKER = range(5, 11)
SELL_CONS = (11, 12)
USE_CONS = (13, 14)
REROLL = 15
SKIP = 16

ACTION_NAMES = (
    ["buy_card_0", "buy_card_1", "buy_pack_0", "buy_pack_1", "buy_voucher"]
    + [f"sell_joker_{j}" for j in range(6)]
    + ["sell_cons_0", "sell_cons_1", "use_cons_0", "use_cons_1", "reroll", "skip"]
)


def to_factored(a: int) -> FactoredAction:
    """Map a shop action index to a Jackdaw action.

    Shop entity targets index each sub-list (shop_cards / shop_boosters /
    shop_vouchers) separately, so the slot number is the target directly.
    """
    if a in BUY_CARD:
        return FactoredAction(AT.BuyCard, entity_target=a - BUY_CARD[0])
    if a in BUY_PACK:
        return FactoredAction(AT.OpenBooster, entity_target=a - BUY_PACK[0])
    if a == BUY_VOUCHER:
        return FactoredAction(AT.RedeemVoucher, entity_target=0)
    if a in SELL_JOKER:
        return FactoredAction(AT.SellJoker, entity_target=a - SELL_JOKER[0])
    if a in SELL_CONS:
        return FactoredAction(AT.SellConsumable, entity_target=a - SELL_CONS[0])
    if a in USE_CONS:
        return FactoredAction(AT.UseConsumable, entity_target=a - USE_CONS[0])
    if a == REROLL:
        return FactoredAction(AT.Reroll)
    if a == SKIP:
        return FactoredAction(AT.NextRound)
    raise ValueError(a)


def shop_mask(mask: ActionMask) -> np.ndarray:
    """Project Jackdaw's factored mask onto the 17 shop actions."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    tm, em = mask.type_mask, mask.entity_masks

    def ok(action_type: int, idx: int) -> bool:
        e = em.get(action_type)
        return bool(tm[action_type] and e is not None and idx < len(e) and e[idx])

    for slot, a in enumerate(BUY_CARD):
        m[a] = ok(AT.BuyCard, slot)
    for slot, a in enumerate(BUY_PACK):
        m[a] = ok(AT.OpenBooster, slot)
    m[BUY_VOUCHER] = ok(AT.RedeemVoucher, 0)
    for slot, a in enumerate(SELL_JOKER):
        m[a] = ok(AT.SellJoker, slot)
    for slot, a in enumerate(SELL_CONS):
        m[a] = ok(AT.SellConsumable, slot)
    # In the shop the engine already masks consumables that need card targets
    for slot, a in enumerate(USE_CONS):
        m[a] = ok(AT.UseConsumable, slot)
    m[REROLL] = bool(tm[AT.Reroll])
    m[SKIP] = bool(tm[AT.NextRound])
    return m


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class ShopAgent(Protocol):
    def act(self, state: np.ndarray, mask: np.ndarray) -> int: ...

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None: ...


class RandomShopAgent:
    """Uniform over legal shop actions: the performance floor."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def act(self, state: np.ndarray, mask: np.ndarray) -> int:
        return self.rng.choice(np.flatnonzero(mask).tolist())

    def observe(self, *transition: Any) -> None:
        pass  # nothing to learn


class RandomPlayAgent:
    """Handles every non-shop phase at random.

    Always selects the blind (never skips) and cashes out.  In a round it
    plays or discards a random 1-5 card subset; in a pack it picks a random
    legal card or skips.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def act(self, gs: dict[str, Any], mask: ActionMask, env: BalatroEnvironment) -> FactoredAction:
        phase = gs["phase"]
        tm = mask.type_mask
        if phase == GamePhase.BLIND_SELECT:
            return FactoredAction(AT.SelectBlind)
        if phase == GamePhase.ROUND_EVAL:
            return FactoredAction(AT.CashOut)
        if phase == GamePhase.SELECTING_HAND:
            options = [at for at in (AT.PlayHand, AT.Discard) if tm[at]]
            at = self.rng.choice(options)
            cards = np.flatnonzero(mask.card_mask).tolist()
            k = self.rng.randint(mask.min_card_select, min(mask.max_card_select, len(cards)))
            return FactoredAction(at, card_target=tuple(sorted(self.rng.sample(cards, k))))
        if phase == GamePhase.PACK_OPENING:
            picks = []
            if tm[AT.PickPackCard]:
                picks = np.flatnonzero(mask.entity_masks[AT.PickPackCard]).tolist()
            choice = self.rng.choice(picks + [None])
            if choice is None:
                return FactoredAction(AT.SkipPack)
            return FactoredAction(AT.PickPackCard, entity_target=choice)
        raise RuntimeError(f"play agent called in phase {phase}")


class SmartPlayAgent:
    """Jackdaw's greedy smart_agent for non-shop phases (for comparison)."""

    def act(self, gs: dict[str, Any], mask: ActionMask, env: BalatroEnvironment) -> FactoredAction:
        from jackdaw.cli.smart_agent import smart_agent
        from jackdaw.engine.actions import PickPackCard

        # get_legal_actions() offers pack picks that pack_pick_block_reason()
        # rejects (e.g. Wheel of Fortune with no eligible joker); the mask
        # doesn't, so drop picks the mask disallows.
        picks = mask.entity_masks.get(AT.PickPackCard)
        legal = [
            a
            for a in env._adapter.get_legal_actions()
            if not isinstance(a, PickPackCard) or (picks is not None and picks[a.card_index])
        ]
        return engine_action_to_factored(smart_agent(gs, legal), gs)


# ---------------------------------------------------------------------------
# ShopEnv
# ---------------------------------------------------------------------------


def r_ante(prev_ante: int, curr_ante: int, alpha: float = 1.0) -> float:
    """alpha * k when ante k is cleared, else 0."""
    return alpha * prev_ante if curr_ante > prev_ante else 0.0


def shop_item_key(gs: dict[str, Any], a: int) -> tuple[str | None, int | None]:
    """(center_key, cost) of the item shop action ``a`` targets, read before stepping."""
    if a in BUY_CARD:
        items, slot = gs.get("shop_cards", []), a - BUY_CARD[0]
    elif a in BUY_PACK:
        items, slot = gs.get("shop_boosters", []), a - BUY_PACK[0]
    elif a == BUY_VOUCHER:
        items, slot = gs.get("shop_vouchers", []), 0
    elif a in SELL_JOKER:
        items, slot = gs.get("jokers", []), a - SELL_JOKER[0]
    elif a in SELL_CONS:
        items, slot = gs.get("consumables", []), a - SELL_CONS[0]
    elif a in USE_CONS:
        items, slot = gs.get("consumables", []), a - USE_CONS[0]
    else:
        return None, None
    if slot >= len(items):
        return None, None
    return items[slot].center_key, items[slot].cost


class RunLogger:
    """Writes steps.jsonl (one line per shop decision), rounds.jsonl (one line
    per blind played, with the cash-out breakdown) and shops.jsonl (one line
    per shop offer: on entering a shop and after each reroll)."""

    def __init__(self, out_dir: Path) -> None:
        self._files = {
            name: open(out_dir / f"{name}.jsonl", "w") for name in ("steps", "rounds", "shops")
        }

    def log_step(self, **row: Any) -> None:
        self._files["steps"].write(json.dumps(row) + "\n")

    def log_round(self, **row: Any) -> None:
        self._files["rounds"].write(json.dumps(row) + "\n")

    def log_shop(self, **row: Any) -> None:
        self._files["shops"].write(json.dumps(row) + "\n")

    def close(self) -> None:
        for f in self._files.values():
            f.close()


SHOP_AREAS = (("card", "shop_cards"), ("voucher", "shop_vouchers"), ("pack", "shop_boosters"))
EARNINGS = {
    "earn_blind": "blind_reward",
    "earn_hands": "unused_hands_bonus",
    "earn_discards": "unused_discards_bonus",
    "earn_interest": "interest",
    "earn_jokers": "joker_dollars",
    "earn_rent": "rental_cost",
    "earn_total": "total",
}


class ShopEnv:
    """Exposes only shop decisions; the play agent drives everything else.

    State is Jackdaw's 235-dim ``global_context`` for now; swap in the matrix
    or summary encoder later.  Reward is R_ante only (no oracle term yet).
    """

    def __init__(
        self,
        play_agent: Any,
        seed_prefix: str = "TRAIN",
        back_keys: tuple[str, ...] = ("b_red",),
        stakes: tuple[int, ...] = (1,),
        logger: RunLogger | None = None,
    ) -> None:
        self.env = BalatroEnvironment(
            adapter_factory=DirectAdapter,
            seed_prefix=seed_prefix,
            back_keys=list(back_keys),
            stakes=list(stakes),
        )
        self.play_agent = play_agent
        self.logger = logger
        self.episode = -1

    @property
    def gs(self) -> dict[str, Any]:
        return self.env._adapter.raw_state

    def reset(self) -> tuple[np.ndarray, np.ndarray, bool]:
        self.stats: Counter[str] = Counter()
        self.episode += 1
        self.step_idx = 0
        obs, mask, info = self.env.reset()
        self.seed = f"{self.env._seed_prefix}_{self.episode}"  # BalatroEnvironment.reset() scheme
        self._obs, self._mask = obs, info["action_mask"]
        done = self._autoplay()
        return self._state(), self._mask17(done), done

    def step(self, a: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict[str, Any]]:
        gs = self.gs
        prev_ante = self.env.episode_ante
        dollars_before = gs["dollars"]
        item_key, item_cost = shop_item_key(gs, a)
        context = {
            "ante": gs["round_resets"]["ante"],
            "round": gs["round"],
            "blind_on_deck": gs.get("blind_on_deck"),
            "jokers_before": [j.center_key for j in gs["jokers"]],
            "consumables_before": [c.center_key for c in gs["consumables"]],
            "legal": np.flatnonzero(shop_mask(self._mask)).tolist(),  # shop action indices
        }

        obs, term, trunc, mask, info = self.env.step(to_factored(a))
        self._obs, self._mask = obs, info["action_mask"]
        self.stats[ACTION_NAMES[a]] += 1
        dollars_after = self.gs["dollars"]
        self.stats["dollars_spent"] += max(0, dollars_before - dollars_after)
        step_idx = self.step_idx
        self.step_idx += 1  # shop offers logged from here on link to the *next* step
        if a == REROLL and self.gs["phase"] == GamePhase.SHOP:
            self._log_shop("reroll")

        done = term or trunc or self._autoplay()
        reward = r_ante(prev_ante, self.env.episode_ante)
        if self.logger:
            self.logger.log_step(
                episode=self.episode,
                step=step_idx,
                **context,
                dollars_before=dollars_before,
                dollars_after=dollars_after,
                action=a,
                action_name=ACTION_NAMES[a],
                item_key=item_key,
                item_cost=item_cost,
                reward=reward,
                done=bool(done),
            )
        info = {"ante": self.env.episode_ante, "won": self.env.episode_won}
        return self._state(), self._mask17(done), reward, done, info

    def _autoplay(self) -> bool:
        """Let the play agent act until the next shop state. Returns done."""
        cashed_out = False
        while self.gs["phase"] != GamePhase.SHOP:
            if self.gs["phase"] == GamePhase.GAME_OVER:
                return True
            was_playing = self.gs["phase"] == GamePhase.SELECTING_HAND
            cashed_out |= self.gs["phase"] == GamePhase.ROUND_EVAL
            ante_played = self.gs["round_resets"]["ante"]
            a = self.play_agent.act(self.gs, self._mask, self.env)
            obs, term, trunc, mask, info = self.env.step(a)
            self._obs, self._mask = obs, info["action_mask"]
            if was_playing and self.gs["phase"] != GamePhase.SELECTING_HAND:
                self._log_round(ante_played)
            if term or trunc:
                return True
        self.stats["shops_visited"] += self.gs["phase"] == GamePhase.SHOP
        if cashed_out:  # a new shop, not a return from a pack opened in the shop
            self._log_shop("enter")
        return False

    def _log_round(self, ante: int) -> None:
        """Record the blind that just ended (chips/blind still hold its values).

        ``ante`` is the ante the blind was played in: beating a boss advances
        ``round_resets.ante`` before we get here.  A beaten blind is in
        ROUND_EVAL with ``round_earnings`` (the cash-out about to be paid); a
        lost one has no earnings.
        """
        gs = self.gs
        blind = gs.get("blind")
        if not self.logger or blind is None:
            return
        beaten = gs["phase"] == GamePhase.ROUND_EVAL
        earnings = gs.get("round_earnings") if beaten else None
        self.logger.log_round(
            episode=self.episode,
            ante=ante,
            round=gs["round"],
            blind_key=blind.key,
            target=blind.chips,
            chips=gs["chips"],
            beaten=gs["phase"] != GamePhase.GAME_OVER,
            dollars_at_end=gs["dollars"],
            interest_cap=gs.get("interest_cap", 25),
            **{col: getattr(earnings, attr, None) for col, attr in EARNINGS.items()},
        )

    def _log_shop(self, reason: str) -> None:
        """Record what the shop offers right now, and at what price."""
        if not self.logger:
            return
        gs = self.gs
        self.logger.log_shop(
            episode=self.episode,
            step=self.step_idx,
            ante=gs["round_resets"]["ante"],
            round=gs["round"],
            reason=reason,
            dollars=gs["dollars"],
            reroll_cost=gs.get("current_round", {}).get("reroll_cost"),
            items=[
                {"kind": kind, "key": c.center_key, "cost": c.cost}
                for kind, area in SHOP_AREAS
                for c in gs.get(area, [])
            ],
        )

    def _state(self) -> np.ndarray:
        return self._obs.global_context

    def _mask17(self, done: bool) -> np.ndarray:
        return np.zeros(N_ACTIONS, dtype=bool) if done else shop_mask(self._mask)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    random.seed(args.seed)  # BalatroEnvironment samples back_keys/stakes from global random
    play_agent = RandomPlayAgent(rng) if args.play == "random" else SmartPlayAgent()
    shop_agent: ShopAgent = RandomShopAgent(rng)

    out_dir = Path(args.out_dir) / (args.run_name or f"random_shop_{args.play}_play")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = None if args.no_step_log else RunLogger(out_dir)
    env = ShopEnv(
        play_agent,
        seed_prefix=args.seed_prefix,
        back_keys=tuple(args.decks),
        stakes=tuple(args.stakes),
        logger=logger,
    )
    write_config(out_dir, args, play_agent, shop_agent)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(out_dir))
    except ImportError:
        print("(tensorboard not installed; logging to CSV only - `uv sync --extra train`)")

    fields = [
        "episode",
        "ante",
        "won",
        "return",
        "shop_steps",
        "shops_visited",
        "dollars_spent",
        "seconds",
        "seed",
        "deck",
        "stake",
        "final_round",
        "final_dollars",
        "final_jokers",
    ] + ACTION_NAMES
    csv_file = open(out_dir / "episodes.csv", "w", newline="")
    log = csv.DictWriter(csv_file, fieldnames=fields)
    log.writeheader()

    total_steps = 0
    antes: list[int] = []
    t_start = time.perf_counter()
    for ep in range(args.episodes):
        t0 = time.perf_counter()
        s, m, done = env.reset()
        ep_return, ep_steps = 0.0, 0
        info = {"ante": env.env.episode_ante, "won": env.env.episode_won}
        while not done:
            a = shop_agent.act(s, m)
            s2, m2, r, done, info = env.step(a)
            shop_agent.observe(s, a, r, s2, done, m2)
            s, m = s2, m2
            ep_return += r
            ep_steps += 1
            total_steps += 1

        antes.append(info["ante"])
        row = {
            "episode": ep,
            "ante": info["ante"],
            "won": int(info["won"]),
            "return": ep_return,
            "shop_steps": ep_steps,
            "shops_visited": env.stats["shops_visited"],
            "dollars_spent": env.stats["dollars_spent"],
            "seconds": round(time.perf_counter() - t0, 4),
            "seed": env.seed,
            "deck": env.gs["selected_back_key"],
            "stake": env.gs["stake"],
            "final_round": env.gs["round"],
            "final_dollars": env.gs["dollars"],
            "final_jokers": ";".join(j.center_key for j in env.gs["jokers"]),
            **{name: env.stats[name] for name in ACTION_NAMES},
        }
        log.writerow(row)
        if writer:
            for key in ("ante", "won", "return", "shop_steps", "dollars_spent"):
                writer.add_scalar(f"episode/{key}", row[key], total_steps)
        if (ep + 1) % args.log_every == 0:
            recent = antes[-args.log_every :]
            print(
                f"ep {ep + 1:5d}  steps {total_steps:7d}  "
                f"mean ante (last {len(recent)}) {np.mean(recent):.2f}  max {max(recent)}"
            )

    csv_file.close()
    if logger:
        logger.close()
    if writer:
        writer.close()
    dist = Counter(antes)
    print(
        f"\n{args.episodes} episodes, {total_steps} shop steps, "
        f"{time.perf_counter() - t_start:.1f}s"
    )
    print(
        f"mean ante {np.mean(antes):.2f}; ante reached: "
        + ", ".join(f"{k}: {dist[k]}" for k in sorted(dist))
    )
    print(f"logs: {out_dir}/")


def write_config(out_dir: Path, args: argparse.Namespace, play_agent: Any, shop_agent: Any) -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except OSError:
        commit = ""
    config = {
        **vars(args),
        "play_agent": type(play_agent).__name__,
        "shop_agent": type(shop_agent).__name__,
        "n_actions": N_ACTIONS,
        "action_names": ACTION_NAMES,
        # 2: rounds.jsonl ante = ante the blind was played in (v1 logged beaten
        # bosses under the next ante), cash-out breakdown, shops.jsonl
        # 3: steps.jsonl legal (shop action indices legal at each decision)
        "log_version": 3,
        "started": datetime.now().isoformat(timespec="seconds"),
        "jackdaw_commit": commit or None,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument(
        "--play",
        choices=["random", "smart"],
        default="random",
        help="non-shop policy: random, or Jackdaw's greedy smart_agent",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-prefix", default="TRAIN")
    ap.add_argument(
        "--decks",
        type=lambda s: s.split(","),
        default=["b_red"],
        help="comma-separated deck keys, sampled per episode (e.g. b_red,b_blue,b_black)",
    )
    ap.add_argument(
        "--stakes",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1],
        help="comma-separated stakes 1-8, sampled per episode",
    )
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default=None, help="default: random_shop_<play>_play")
    ap.add_argument("--no-step-log", action="store_true", help="skip steps.jsonl / rounds.jsonl")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
