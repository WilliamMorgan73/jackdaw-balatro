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


def test_gym_actions_unique_under_forced_card():
    """Each gym play/discard slot is a distinct engine action that includes the forced card."""
    from jackdaw.env import DirectAdapter
    from jackdaw.env.balatro_env import _action_mask_to_game
    from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

    env = BalatroGymnasiumEnv(DirectAdapter, back_keys=["b_red"], stakes=[1])
    env.reset(seed=7)
    select = next(
        i for i, a in enumerate(env._action_table) if a.action_type == ActionType.SelectBlind
    )
    env.step(select)
    gs = env._inner._adapter.raw_state
    gs["hand"][2].ability["forced_selection"] = True
    mask = _action_mask_to_game(env._inner._action_mask(gs))
    table = env._enumerate_actions(mask, {"raw_state": gs})
    for at in (ActionType.PlayHand, ActionType.Discard):
        engine = [factored_to_engine_action(a, gs) for a in table if a.action_type == at]
        assert engine
        assert len(engine) == len(set(engine))
        assert all(2 in a.card_indices for a in engine)
