"""Cerulean Bell's forced card is added to factored play/discard selections."""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import Discard, PlayHand, SelectBlind
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.action_space import ActionType, FactoredAction, factored_to_engine_action


@pytest.fixture
def gs():
    gs = initialize_run("b_red", 1, "FORCED")
    step(gs, SelectBlind())
    gs["hand"][2].ability["forced_selection"] = True
    return gs


@pytest.mark.parametrize(
    ("chosen", "expected"),
    [
        ((0, 1), (0, 1, 2)),  # forced card added
        ((0, 2, 4), (0, 2, 4)),  # already included
        ((0, 1, 3, 4, 5), (0, 1, 2, 3, 4)),  # 5 + forced: the last requested card drops
    ],
)
@pytest.mark.parametrize(
    ("action_type", "engine_type"),
    [(ActionType.PlayHand, PlayHand), (ActionType.Discard, Discard)],
)
def test_forced_card_added(gs, chosen, expected, action_type, engine_type):
    action = factored_to_engine_action(FactoredAction(action_type, card_target=chosen), gs)
    assert action == engine_type(card_indices=expected)


def test_converted_action_executes(gs):
    with pytest.raises(IllegalActionError):
        step(gs, PlayHand(card_indices=(0, 1)))
    step(gs, factored_to_engine_action(FactoredAction(ActionType.PlayHand, card_target=(0, 1)), gs))


def test_no_forced_card_unchanged():
    gs = initialize_run("b_red", 1, "FORCED")
    step(gs, SelectBlind())
    action = factored_to_engine_action(FactoredAction(ActionType.PlayHand, card_target=(3, 1)), gs)
    assert action == PlayHand(card_indices=(3, 1))
