"""The Inflation challenge: every shop purchase raises all prices by $1.

Vanilla (c_inflation_1, custom rule 'inflation'): buy_from_shop does
``G.GAME.inflation += 1`` and re-runs set_cost on every card.
"""

from __future__ import annotations

from jackdaw.engine.actions import BuyCard, CashOut, PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.challenges import CHALLENGES
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run


def shop_state(seed, **run_kw):
    """A run advanced to the first shop (Small Blind beaten)."""
    gs = initialize_run("b_red", 1, seed, **run_kw)
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    step(gs, CashOut())
    return gs


def _inflation_shop():
    gs = shop_state("INFLATION", challenge=CHALLENGES["c_inflation_1"])
    a, b = create_joker("j_joker"), create_joker("j_greedy_joker")
    a.set_cost()
    b.set_cost()
    gs["shop_cards"] = [a, b]
    gs["jokers"] = []
    gs["dollars"] = 100
    return gs, b


def test_purchase_increments_inflation():
    gs, _ = _inflation_shop()
    step(gs, BuyCard(shop_index=0))
    assert gs["inflation"] == 1


def test_purchase_raises_remaining_shop_prices():
    gs, other = _inflation_shop()
    cost_before = other.cost
    step(gs, BuyCard(shop_index=0))
    assert other.cost == cost_before + 1
