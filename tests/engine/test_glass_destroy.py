"""Card destruction after scoring matches vanilla's evaluate_play loop.

Vanilla evaluate_play destroy loop: every scoring card is offered to the
jokers' ``destroying_card`` (no debuff gate), then
``pseudorandom('glass')`` is rolled for every non-debuffed Glass card
even if a joker already destroyed it.
"""

from __future__ import annotations

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand


def _six(enhancement: str = "c_base") -> Card:
    c = Card()
    c.set_base("H_6", "Hearts", "6")
    c.set_ability(enhancement)
    return c


def _score(card: Card, jokers: list[Card], rng: PseudoRandom):
    return score_hand([card], [], jokers, HandLevels(), Blind.create("bl_small", ante=1), rng)


def test_glass_roll_consumed_even_when_sixth_sense_destroys():
    rng = PseudoRandom("GLASS6")
    glass_six = _six("m_glass")
    result = _score(glass_six, [create_joker("j_sixth_sense")], rng)
    assert glass_six in result.cards_destroyed
    # The 'glass' stream must have advanced, as in vanilla.
    assert "glass" in rng.state


def test_sixth_sense_destroys_debuffed_six():
    six = _six()
    six.debuff = True
    result = _score(six, [create_joker("j_sixth_sense")], PseudoRandom("DEBUFF6"))
    assert six in result.cards_destroyed
