"""Rent at round end: charged once, with no debuff gate, and reported in the
cash-out breakdown.

``round_lifecycle.process_round_end_cards`` charges the rent (vanilla
``calculate_rental``, card.lua:2271, has no debuff gate).
``gs["round_earnings"].rental_cost`` reports that charge and must not deduct
it a second time.
"""

from __future__ import annotations

from jackdaw.engine.actions import CashOut, PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run


def _run_round(*, rental: bool, debuffed: bool = False) -> dict:
    gs = initialize_run("b_red", 1, "RENT_E2E")
    joker = create_joker("j_joker", rental=rental)
    gs["jokers"] = [joker]
    joker.add_to_deck(gs)
    step(gs, SelectBlind())
    joker.debuff = debuffed
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    step(gs, CashOut())
    return gs


def test_rent_is_charged():
    assert _run_round(rental=False)["dollars"] - _run_round(rental=True)["dollars"] == 3


def test_rent_is_charged_on_debuffed_joker():
    base = _run_round(rental=False, debuffed=True)["dollars"]
    assert base - _run_round(rental=True, debuffed=True)["dollars"] == 3


def test_round_earnings_reports_rent():
    assert _run_round(rental=True)["round_earnings"].rental_cost == 3
