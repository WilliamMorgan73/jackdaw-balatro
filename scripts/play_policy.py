"""A stronger play policy than Jackdaw's ``smart_agent``, for the shop-only harness.

Only playing and discarding change.  Blind select, cash-out, packs and
consumables still come from ``smart_agent``, so a comparison against
``--play smart`` changes one thing.  What's different:

- **Exact scores.** The top ``k`` candidate hands (ranked by
  ``smart_agent``'s fast estimate) are scored by the engine's own
  ``score_hand`` on a copy of the hand, jokers, hand levels and blind.  So
  boss debuffs (The Psychic, The Eye, The Mouth, ...), every joker and card
  enhancements count.  The copy draws from a fresh RNG, so the policy can't
  see whether a Lucky Card will actually trigger.
- **Blind-aware discards.** Play the best hand if it beats what the blind
  still needs, or if no discards or only one hand are left.  Discard only
  when playing the best hand every remaining turn can't win.
- **Draw-aware discards.** Keep the flush draw, straight draw, pairs or best
  hand that is worth most (chance of completing it from the cards still in
  the deck, which the player can see, times its base score) and throw the
  rest.

Limitations: ``_press_play`` boss effects (The Hook, The Tooth) aren't
simulated, and, like ``smart_agent``, it reads face-down hand cards.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from itertools import combinations
from math import comb
from typing import Any

from jackdaw.cli.smart_agent import _estimate_hand_score, smart_agent
from jackdaw.engine.actions import Discard, GamePhase, PlayHand, UseConsumable
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand
from jackdaw.env.action_space import ActionMask, FactoredAction, engine_action_to_factored

SUITS = ("Spades", "Hearts", "Clubs", "Diamonds")
# Straight windows by rank id (2..14); the ace also plays low.
WINDOWS = [tuple(range(lo, lo + 5)) for lo in range(2, 11)] + [(14, 2, 3, 4, 5)]
AVG_CHIPS = 8  # chips a card still to be drawn is assumed to add


# ---------------------------------------------------------------------------
# Draw odds
# ---------------------------------------------------------------------------


def hypergeom_at_least(n_deck: int, outs: int, draws: int, need: int) -> float:
    """P(at least ``need`` of ``outs`` cards in ``draws`` draws from ``n_deck``)."""
    if need <= 0:
        return 1.0
    draws = min(draws, n_deck)
    if draws < need or outs < need:
        return 0.0
    total = comb(n_deck, draws)
    miss = sum(comb(outs, i) * comb(n_deck - outs, draws - i) for i in range(need))
    return 1.0 - miss / total


def p_all_ranks(n_deck: int, outs: Sequence[int], draws: int) -> float:
    """P(at least one card of every missing rank in ``draws`` draws), where
    ``outs[i]`` is how many cards of missing rank i are left in the deck."""
    draws = min(draws, n_deck)
    if len(outs) > draws:
        return 0.0
    total = comb(n_deck, draws)
    p = 0.0
    for r in range(len(outs) + 1):  # inclusion-exclusion over ranks drawn none of
        for subset in combinations(outs, r):
            p += (-1) ** r * comb(n_deck - sum(subset), draws) / total
    return max(p, 0.0)


# ---------------------------------------------------------------------------
# Discard targets
# ---------------------------------------------------------------------------


def suit_groups(cards: Sequence[Any]) -> dict[str, list[int]]:
    """Hand indices that count towards a flush in each suit (Wild in all)."""
    return {s: [i for i, c in enumerate(cards) if c.is_suit(s, flush_calc=True)] for s in SUITS}


def straight_window(cards: Sequence[Any]) -> tuple[tuple[int, ...], list[int]]:
    """The 5-rank window holding the most distinct ranks in hand: the window,
    and one hand index per rank it holds."""
    best: tuple[tuple[int, ...], list[int]] = ((), [])
    for window in WINDOWS:
        picked: dict[int, int] = {}
        for i, c in enumerate(cards):
            r = c.get_id()
            if r in window and r not in picked:
                picked[r] = i
        if len(picked) > len(best[1]):
            best = (window, sorted(picked.values()))
    return best


def _base_value(gs: dict[str, Any], hand_type: str, kept: Sequence[Any], missing: int) -> float:
    """Level-scaled base score of ``hand_type`` made from ``kept`` plus
    ``missing`` cards still to be drawn (no jokers: compares targets)."""
    chips, mult = gs["hand_levels"].get(hand_type)
    card_chips = sum(c.get_chip_bonus() for c in kept) + AVG_CHIPS * missing
    return (chips + card_chips) * mult


def discard_targets(gs: dict[str, Any], best: Sequence[int], draws: int) -> list[tuple]:
    """Candidate cards to keep, each as ``(value, keep indices)``."""
    hand, deck = gs["hand"], gs["deck"]
    n = len(deck)
    best_type, _ = _estimate_hand_score([hand[i] for i in best], gs["jokers"], gs["hand_levels"])
    targets = [(_base_value(gs, best_type, [hand[i] for i in best], 0), list(best))]

    suit, keep = max(suit_groups(hand).items(), key=lambda kv: len(kv[1]))
    if 3 <= len(keep) < 5:
        outs = sum(c.is_suit(suit, flush_calc=True) for c in deck)
        p = hypergeom_at_least(n, outs, draws, 5 - len(keep))
        kept = [hand[i] for i in keep]
        targets.append((p * _base_value(gs, "Flush", kept, 5 - len(keep)), keep))

    window, keep = straight_window(hand)
    if 3 <= len(keep) < 5:
        held = {hand[i].get_id() for i in keep}
        outs = [sum(c.get_id() == r for c in deck) for r in window if r not in held]
        p = p_all_ranks(n, outs, draws)
        kept = [hand[i] for i in keep]
        targets.append((p * _base_value(gs, "Straight", kept, len(outs)), keep))

    ranks: dict[int, list[int]] = {}
    for i, c in enumerate(hand):
        if c.get_id() > 0:
            ranks.setdefault(c.get_id(), []).append(i)
    groups = sorted((g for g in ranks.values() if len(g) >= 2), key=len, reverse=True)
    if groups:
        keep = sorted(i for g in groups[:2] for i in g)[:5]
        kept = [hand[i] for i in keep]
        made, _ = _estimate_hand_score(kept, gs["jokers"], gs["hand_levels"])
        targets.append((_base_value(gs, made, kept, 0), keep))
    return targets


def choose_discard(gs: dict[str, Any], best: Sequence[int], max_cards: int = 5) -> list[int]:
    """Hand indices to discard: everything outside the most valuable target,
    lowest-chip cards first, at most ``max_cards``.  Empty if nothing to throw."""
    hand = gs["hand"]
    draws = min(max_cards, len(hand))
    _, keep = max(discard_targets(gs, best, draws), key=lambda t: t[0])
    rest = sorted(
        (i for i in range(len(hand)) if i not in keep), key=lambda i: hand[i].get_chip_bonus()
    )
    return sorted(rest[:max_cards])


# ---------------------------------------------------------------------------
# Exact scoring
# ---------------------------------------------------------------------------


def exact_score(gs: dict[str, Any], indices: Sequence[int]) -> float:
    """What playing ``indices`` would score, from the engine's ``score_hand``.

    Runs on a deep copy of the hand, jokers, hand levels, blind and
    consumables (scoring mutates them: The Eye, scaling jokers, play counts)
    with a fresh RNG, and fills the snapshot fields the way ``play_hand``
    does (jackdaw/engine/game.py).  ``gs`` is left untouched.
    """
    hand, jokers, levels, blind, consumables = copy.deepcopy(
        (gs["hand"], gs["jokers"], gs["hand_levels"], gs["blind"], gs["consumables"])
    )
    picked = set(indices)
    played = [hand[i] for i in indices]
    held = [c for i, c in enumerate(hand) if i not in picked]
    for c in played:
        c.base.times_played = getattr(c.base, "times_played", 0) + 1
        c.ability["played_this_ante"] = True
    blind.triggered = False

    cr = gs["current_round"]
    all_cards = gs["deck"] + gs["hand"] + gs["discard_pile"]
    usage = gs.get("consumable_usage_total", {})
    snap = dict(
        gs,
        hand=held,
        hands_left=cr.get("hands_left", 0) - 1,
        current_round_hands_played=cr.get("hands_played", 0),
        discards_left=cr.get("discards_left", 0),
        discards_used=cr.get("discards_used", 0),
        money=gs.get("dollars", 0),
        deck_cards_remaining=len(gs["deck"]),
        playing_cards_count=len(all_cards),
        stone_tally=sum(c.center_key == "m_stone" for c in all_cards),
        steel_tally=sum(c.center_key == "m_steel" for c in all_cards),
        enhanced_card_count=sum(c.center_key not in ("", "c_base") for c in all_cards),
        mail_card_id=cr.get("mail_card", {}).get("id"),
        idol_card=cr.get("idol_card"),
        ancient_suit=cr.get("ancient_card", {}).get("suit"),
        consumable_usage_tarot=usage.get("tarot", 0),
    )
    seed = f"play_policy_{gs['rng'].seed_str}_{gs['round']}_{cr.get('hands_played', 0)}"
    result = score_hand(
        played_cards=played,
        held_cards=held,
        jokers=jokers,
        hand_levels=levels,
        blind=blind,
        rng=PseudoRandom(seed),
        probabilities_normal=gs.get("probabilities", {}).get("normal", 1),
        game_state=snap,
        back_key=gs.get("selected_back_key"),
        blind_chips=blind.chips,
        consumables=consumables,
    )
    return result.total


def candidates(gs: dict[str, Any], k: int) -> list[tuple[int, ...]]:
    """The ``k`` best 1-5 card subsets by ``smart_agent``'s estimate, plus the
    best 5-card ones (The Psychic scores nothing smaller)."""
    hand = gs["hand"]
    scored = []
    for size in range(1, min(5, len(hand)) + 1):
        for combo in combinations(range(len(hand)), size):
            _, est = _estimate_hand_score([hand[i] for i in combo], gs["jokers"], gs["hand_levels"])
            scored.append((est, size, combo))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    picks = [combo for _, _, combo in scored[:k]]
    picks += [combo for _, size, combo in scored if size == 5][:2]
    return list(dict.fromkeys(picks))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _uses_planet(gs: dict[str, Any], legal_actions: list[Any]) -> bool:
    """Whether ``smart_agent`` would use a held planet before playing."""
    cons = gs.get("consumables", [])
    return any(
        isinstance(a, UseConsumable)
        and a.card_index < len(cons)
        and cons[a.card_index].ability.get("set") == "Planet"
        for a in legal_actions
    )


def strong_agent(gs: dict[str, Any], legal_actions: list[Any], k: int = 8) -> Any:
    """``smart_agent`` with exact, blind-aware, draw-aware play and discard."""
    if (
        gs["phase"] != GamePhase.SELECTING_HAND
        or not gs.get("hand")
        or _uses_planet(gs, legal_actions)
    ):
        return smart_agent(gs, legal_actions)

    scores = {c: exact_score(gs, c) for c in candidates(gs, k)}
    best = max(scores, key=lambda c: (scores[c], len(c)))
    cr = gs["current_round"]
    hands_left, discards_left = cr.get("hands_left", 1), cr.get("discards_left", 0)
    need = gs["blind"].chips - gs["chips"]
    can_discard = discards_left > 0 and any(isinstance(a, Discard) for a in legal_actions)
    if can_discard and hands_left > 1 and scores[best] < need and scores[best] * hands_left < need:
        throw = choose_discard(gs, best)
        if throw:
            return Discard(card_indices=tuple(throw))
    return PlayHand(card_indices=tuple(best))


class StrongPlayAgent:
    """``strong_agent`` for every non-shop phase (``--play strong``)."""

    def __init__(self, k: int = 8) -> None:
        self.k = k

    @property
    def params(self) -> dict[str, int]:
        return {"k": self.k}

    def act(self, gs: dict[str, Any], mask: ActionMask, env: Any) -> FactoredAction:
        return engine_action_to_factored(
            strong_agent(gs, env._adapter.get_legal_actions(), self.k), gs
        )
