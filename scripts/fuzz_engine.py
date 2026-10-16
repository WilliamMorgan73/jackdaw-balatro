"""Invariant fuzzer for the engine and the env action mask.

Plays full runs over every deck and stake with a mix of Jackdaw's smart_agent,
random engine legal actions and the env's RandomAgent (sampling from the mask).
Every action goes through the env's factored conversion, as BalatroEnvironment
does.  After every step it checks:

- the chosen action executes (no IllegalActionError, no crash);
- the mask and get_legal_actions agree (types and entity slots);
- joker / consumable slots, a card never in two areas, no negative shop
  prices, owned cards sell for at least $1;
- a purchase never takes money below ``bankrupt_at``;
- cash-out pays exactly the ``round_earnings`` breakdown, plus $25 per
  Investment tag that pays out.

Money may legitimately go negative without Credit Card (rent, The Tooth),
so that is not flagged.  ``--easy`` sets every blind to 1 chip, so runs reach
ante 8 and endless mode and exercise late-game content.

Usage::

    uv run python scripts/fuzz_engine.py --games 300
    uv run python scripts/fuzz_engine.py --games 150 --easy --p-smart 0.2

Exits 1 if any invariant failed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.cli.smart_agent import smart_agent
from jackdaw.engine.actions import CashOut, Discard, GamePhase, PlayHand, get_legal_actions
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.action_space import ActionType as AT
from jackdaw.env.action_space import (
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents import RandomAgent

CENTERS = Path(__file__).parent.parent / "jackdaw" / "engine" / "data" / "centers.json"
BACKS = sorted(
    k for k in json.loads(CENTERS.read_text()) if k.startswith("b_") and k != "b_challenge"
)
SPENDING = ("BuyCard", "Reroll", "RedeemVoucher", "OpenBooster")
INVESTMENT_DOLLARS = 25
EXAMPLES = 5


class Report:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def __call__(self, kind: str, detail: str, ctx: dict[str, Any]) -> None:
        self.counts[kind] += 1
        if len(self.examples[kind]) < EXAMPLES:
            self.examples[kind].append({"detail": detail, **ctx})


def _negative(card: Any) -> bool:
    return bool(getattr(card, "edition", None) and card.edition.get("negative"))


def check_mask(gs: dict, legal: list, mask: Any, note: Report, ctx: dict) -> None:
    by_type: dict[int, set] = defaultdict(set)
    for a in legal:
        try:
            fa = engine_action_to_factored(a, gs)
        except Exception as e:  # noqa: BLE001
            note("legal action not convertible", f"{a!r}: {e!r}", ctx)
            continue
        at = int(fa.action_type)
        by_type[at].add(fa.entity_target)
        if not mask.type_mask[at]:
            note("legal action type masked out", f"{AT(at).name} {a!r}", ctx)
        elif fa.entity_target is not None and at in mask.entity_masks:
            em = mask.entity_masks[at]
            if fa.entity_target >= len(em) or not em[fa.entity_target]:
                note("legal entity masked out", f"{AT(at).name} {a!r}", ctx)
    for at in map(int, np.flatnonzero(mask.type_mask)):
        if at not in by_type:
            note("mask allows a type with no legal action", AT(at).name, ctx)
        elif at in mask.entity_masks and None not in by_type[at]:
            extra = set(np.flatnonzero(mask.entity_masks[at]).tolist()) - by_type[at]
            if extra:
                note(
                    "mask allows an entity not in legal actions",
                    f"{AT(at).name} {sorted(extra)}",
                    ctx,
                )


def check_state(gs: dict, note: Report, ctx: dict) -> None:
    jokers, cons = gs.get("jokers", []), gs.get("consumables", [])
    if len(jokers) - sum(map(_negative, jokers)) > gs.get("joker_slots", 5):
        note("too many jokers", f"{len(jokers)} > {gs.get('joker_slots')}", ctx)
    if len(cons) - sum(map(_negative, cons)) > gs.get("consumable_slots", 2):
        note("too many consumables", f"{len(cons)} > {gs.get('consumable_slots')}", ctx)
    seen: dict[int, str] = {}
    for area in ("deck", "hand", "discard_pile", "played_cards_area", "jokers", "consumables"):
        for c in gs.get(area, []):
            if id(c) in seen:
                note(
                    "card in two areas",
                    f"{getattr(c, 'center_key', c)}: {seen[id(c)]}, {area}",
                    ctx,
                )
            seen[id(c)] = area
    for c in jokers + cons:
        if getattr(c, "sell_cost", 1) < 1:
            note("sell value < 1", f"{c.center_key} sells for {c.sell_cost}", ctx)
    for area in ("shop_cards", "shop_vouchers", "shop_boosters"):
        for c in gs.get(area, []):
            if getattr(c, "cost", 0) < 0:
                note("negative shop price", f"{c.center_key} costs {c.cost}", ctx)


def _investments_paid(gs: dict) -> int:
    return sum(
        1
        for e in gs.get("awarded_tags", [])
        if e.get("key") == "tag_investment" and e.get("eval_fired")
    )


def choose(
    gs: dict, legal: list, mask: Any, rng: random.Random, agent: RandomAgent, p_smart: float
):
    """(engine action, source); every action goes through the env's conversion."""
    if rng.random() < p_smart:
        action, src = smart_agent(gs, legal), "smart"
    elif rng.random() < 0.5:
        action, src = rng.choice(legal), "legal"
        if isinstance(action, (PlayHand, Discard)) and not action.card_indices:
            # get_legal_actions returns play/discard as markers: pick the cards
            k = rng.randint(1, min(5, len(gs["hand"])))
            action = type(action)(card_indices=tuple(sorted(rng.sample(range(len(gs["hand"])), k))))
    else:
        return factored_to_engine_action(agent.act({}, mask, {"raw_state": gs}), gs), "mask"
    return factored_to_engine_action(engine_action_to_factored(action, gs), gs), src


def play(seed: str, back: str, stake: int, rng: random.Random, args, note: Report) -> dict:
    gs = initialize_run(back, stake, seed)
    agent = RandomAgent()
    for t in range(args.max_steps):
        ante = gs["round_resets"]["ante"]
        if gs.get("phase") == GamePhase.GAME_OVER or (gs.get("won") and not args.easy):
            return gs
        if args.easy and ante > args.max_ante:
            return gs
        if args.easy and gs.get("phase") == GamePhase.SELECTING_HAND and gs.get("blind"):
            gs["blind"].chips = 1
        ctx = {
            "seed": seed,
            "back": back,
            "stake": stake,
            "step": t,
            "phase": str(gs.get("phase")),
            "ante": ante,
        }
        legal = get_legal_actions(gs)
        mask = get_action_mask(gs)
        check_mask(gs, legal, mask, note, ctx)
        if not legal:
            note("no legal actions", "", ctx)
            return gs
        try:
            action, src = choose(gs, legal, mask, rng, agent, args.p_smart)
        except Exception as e:  # noqa: BLE001
            note("agent / conversion error", repr(e), ctx)
            return gs
        before = gs.get("dollars", 0)
        breakdown = getattr(gs.get("round_earnings"), "total", None)
        invested = _investments_paid(gs)
        try:
            step(gs, action)
        except IllegalActionError as e:
            note(f"illegal action ({src})", f"{action!r}: {e}", ctx)
            return gs
        except Exception as e:  # noqa: BLE001
            note("crash", f"{action!r}: {e!r} | {traceback.format_exc(limit=-3)}", ctx)
            return gs
        if isinstance(action, CashOut) and breakdown is not None:
            tags = (_investments_paid(gs) - invested) * INVESTMENT_DOLLARS
            if gs["dollars"] - before != breakdown + tags:
                note(
                    "cash-out != breakdown + tags",
                    f"paid {gs['dollars'] - before}, breakdown {breakdown}, tags {tags}",
                    ctx,
                )
        if type(action).__name__ in SPENDING and gs.get("dollars", 0) < min(
            before, gs.get("bankrupt_at", 0)
        ):
            note("purchase below bankrupt_at", f"{action!r}: ${before} -> ${gs['dollars']}", ctx)
        check_state(gs, note, ctx)
    note("step cap reached", "", {"seed": seed, "back": back, "stake": stake})
    return gs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--start", type=int, default=0, help="first game index (seed FUZZ<i>)")
    ap.add_argument("--p-smart", type=float, default=0.75, help="share of smart_agent actions")
    ap.add_argument("--easy", action="store_true", help="1-chip blinds: reach ante 8 and endless")
    ap.add_argument("--max-ante", type=int, default=12, help="with --easy, stop after this ante")
    ap.add_argument("--max-steps", type=int, default=6000)
    args = ap.parse_args()

    rng = random.Random(args.start)
    random.seed(args.start)  # RandomAgent samples from the global RNG
    note = Report()
    antes: Counter[int] = Counter()
    for i in range(args.start, args.start + args.games):
        back, stake = BACKS[i % len(BACKS)], 1 + (i // len(BACKS)) % 8
        antes[play(f"FUZZ{i}", back, stake, rng, args, note)["round_resets"]["ante"]] += 1
    print("antes reached:", ", ".join(f"{a}: {n}" for a, n in sorted(antes.items())))
    if not note.counts:
        print("no invariant failures")
        return
    for kind, n in note.counts.most_common():
        print(f"\n{kind}: {n}")
        for ex in note.examples[kind]:
            print("   ", json.dumps(ex, default=str))
    sys.exit(1)


if __name__ == "__main__":
    main()
