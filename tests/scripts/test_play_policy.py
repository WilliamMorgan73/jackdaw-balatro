"""The stronger play policy in scripts/play_policy.py."""

from __future__ import annotations

import copy
import json
import sys
from math import comb
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"


@pytest.fixture(scope="module")
def pp():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import play_policy
    finally:
        sys.path.remove(str(SCRIPTS))
    return play_policy


def _card(rank: str, suit: str, enhancement: str = "c_base"):
    from jackdaw.engine.card_factory import create_playing_card
    from jackdaw.engine.data.enums import Rank, Suit

    return create_playing_card(Suit(suit), Rank(rank), enhancement=enhancement)


def _hand(*codes: str):
    """'AH', '10S', 'KD' ... -> playing cards."""
    suits = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
    ranks = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack"}
    return [_card(ranks.get(c[:-1], c[:-1]), suits[c[-1]]) for c in codes]


@pytest.fixture
def game():
    """A fresh game at its first SELECTING_HAND state."""
    from jackdaw.cli.smart_agent import smart_agent
    from jackdaw.engine.actions import GamePhase
    from jackdaw.env import BalatroEnvironment, DirectAdapter
    from jackdaw.env.action_space import engine_action_to_factored

    env = BalatroEnvironment(adapter_factory=DirectAdapter, seed_prefix="PP")
    env.reset(seed="PP_0")
    gs = env._adapter.raw_state
    while gs["phase"] != GamePhase.SELECTING_HAND:
        legal = env._adapter.get_legal_actions()
        env.step(engine_action_to_factored(smart_agent(gs, legal), gs))
    return env


# -- draw odds ---------------------------------------------------------------


def test_hypergeom_at_least(pp):
    assert pp.hypergeom_at_least(40, 9, 3, 0) == 1.0
    assert pp.hypergeom_at_least(40, 9, 1, 2) == 0.0  # can't draw 2 in 1
    assert pp.hypergeom_at_least(40, 0, 5, 1) == 0.0
    assert pp.hypergeom_at_least(40, 9, 3, 1) == pytest.approx(1 - comb(31, 3) / comb(40, 3))
    assert pp.hypergeom_at_least(4, 4, 10, 4) == 1.0  # draws capped at the deck


def test_p_all_ranks(pp):
    assert pp.p_all_ranks(40, [4], 3) == pytest.approx(pp.hypergeom_at_least(40, 4, 3, 1))
    assert pp.p_all_ranks(40, [4, 4], 1) == 0.0
    assert pp.p_all_ranks(40, [4, 0], 5) == 0.0  # a rank with no cards left
    # 1 - P(no A) - P(no B) + P(neither)
    two = 1 - 2 * comb(36, 5) / comb(40, 5) + comb(32, 5) / comb(40, 5)
    assert pp.p_all_ranks(40, [4, 4], 5) == pytest.approx(two)


# -- discard targets ---------------------------------------------------------


def test_suit_groups_count_wild_cards_in_every_suit(pp):
    hand = _hand("2H", "5H", "KS")
    hand.append(_card("9", "Clubs", enhancement="m_wild"))
    groups = pp.suit_groups(hand)
    assert groups["Hearts"] == [0, 1, 3]
    assert groups["Spades"] == [2, 3]
    assert groups["Diamonds"] == [3]


@pytest.mark.parametrize(
    ("codes", "window", "picked"),
    [
        (["AS", "2H", "3C", "5D", "KS"], (14, 2, 3, 4, 5), [0, 1, 2, 3]),
        (["10S", "JH", "QC", "AD", "2S"], (10, 11, 12, 13, 14), [0, 1, 2, 3]),
        (["7S", "7H", "8C", "9D", "2S"], (5, 6, 7, 8, 9), [0, 2, 3]),
    ],
)
def test_straight_window(pp, codes, window, picked):
    assert pp.straight_window(_hand(*codes)) == (window, picked)


def _discard_state(hand, deck):
    from jackdaw.engine.hand_levels import HandLevels

    return {"hand": hand, "deck": deck, "jokers": [], "hand_levels": HandLevels()}


def test_choose_discard_keeps_a_four_card_flush_draw(pp):
    hand = _hand("2H", "6H", "9H", "KH", "3S", "8C", "4D", "JS")
    deck = _hand(*[f"{r}H" for r in ("3", "4", "5", "7", "8", "10", "Q", "A")], "2S", "5C")
    throw = pp.choose_discard(_discard_state(hand, deck), best=(3,))
    assert throw == [4, 5, 6, 7]


def test_choose_discard_keeps_a_pair_with_no_draws(pp):
    hand = _hand("KH", "KS", "2C", "7D", "9S")
    deck = _hand("3D", "4C", "6C", "8D", "10C", "JD")
    throw = pp.choose_discard(_discard_state(hand, deck), best=(0, 1))
    assert throw == [2, 3, 4]


# -- exact scoring -----------------------------------------------------------


def test_exact_score_matches_the_engine_and_leaves_gs_alone(pp, game):
    from jackdaw.engine.actions import PlayHand
    from jackdaw.engine.rng import PseudoRandom
    from jackdaw.env.action_space import engine_action_to_factored

    gs = game._adapter.raw_state
    rng_state = dict(gs["rng"].state)
    levels = copy.deepcopy(gs["hand_levels"])
    for indices in [(0,), (0, 1), (0, 1, 2, 3, 4)]:
        predicted = pp.exact_score(gs, indices)
        sim = copy.deepcopy(game)
        sgs = sim._adapter.raw_state
        played = sgs["current_round"]["hands_played"]
        sgs["rng"] = PseudoRandom(f"play_policy_{sgs['rng'].seed_str}_{sgs['round']}_{played}")
        action = PlayHand(card_indices=indices)
        sim.step(engine_action_to_factored(action, sgs))
        assert predicted == sim._adapter.raw_state["last_score_result"].total
    assert gs["rng"].state == rng_state
    assert gs["chips"] == 0
    for ht in ("Pair", "High Card", "Flush"):
        assert gs["hand_levels"].get_state(ht).played == levels.get_state(ht).played


# -- decisions ---------------------------------------------------------------


def _decide(pp, game, *, need, hands_left, discards_left, monkeypatch, best=100.0):
    gs = game._adapter.raw_state
    gs["blind"].chips = need
    gs["current_round"]["hands_left"] = hands_left
    gs["current_round"]["discards_left"] = discards_left
    monkeypatch.setattr(pp, "exact_score", lambda gs, c: best if c == (0, 1) else 1.0)
    monkeypatch.setattr(pp, "candidates", lambda gs, k: [(0,), (0, 1)])
    return pp.strong_agent(gs, game._adapter.get_legal_actions())


@pytest.mark.parametrize(
    ("need", "hands_left", "discards_left", "discard"),
    [
        (100, 4, 3, False),  # the best hand already wins
        (250, 4, 3, False),  # playing it every hand wins
        (500, 4, 3, True),  # it can't: chase
        (500, 1, 3, False),  # last hand: play
        (500, 4, 0, False),  # no discards left
    ],
)
def test_play_or_discard(pp, game, monkeypatch, need, hands_left, discards_left, discard):
    from jackdaw.engine.actions import Discard, PlayHand

    a = _decide(
        pp,
        game,
        need=need,
        hands_left=hands_left,
        discards_left=discards_left,
        monkeypatch=monkeypatch,
    )
    assert isinstance(a, Discard if discard else PlayHand)
    if not discard:
        assert a.card_indices == (0, 1)


@pytest.mark.slow
def test_strong_play_only_takes_legal_actions(tmp_path, monkeypatch):
    import importlib.util

    script = SCRIPTS / "train_random_shop.py"
    spec = importlib.util.spec_from_file_location("train_random_shop", script)
    trs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trs
    spec.loader.exec_module(trs)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    args = [
        "--shop", "heuristic", "--play", "strong", "--episodes", "4", "--log-every", "4",
        "--out-dir", str(tmp_path), "--run-name", "strong",
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train_random_shop.py", *args])
    trs.main()  # the env raises on an illegal action
    config = json.loads((tmp_path / "strong" / "config.json").read_text())
    assert config["play_agent"] == "StrongPlayAgent"
    assert config["play_params"] == {"k": 8}
