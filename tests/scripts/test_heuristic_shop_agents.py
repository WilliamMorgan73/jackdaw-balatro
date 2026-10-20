"""The hand-written ceiling-test shops in scripts/train_random_shop.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "train_random_shop.py"


@pytest.fixture(scope="module")
def trs():
    spec = importlib.util.spec_from_file_location("train_random_shop", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass/typing lookups need it registered
    spec.loader.exec_module(module)
    return module


def _mask(trs, *legal: int) -> np.ndarray:
    m = np.zeros(trs.N_ACTIONS, dtype=bool)
    m[list(legal)] = True
    return m


def _card(set_: str, cost: int = 4, **ability) -> SimpleNamespace:
    return SimpleNamespace(ability={"set": set_, "name": "", **ability}, cost=cost)


def _gs(dollars: int, **kw) -> dict:
    return {
        "dollars": dollars,
        "consumables": [],
        "shop_cards": [],
        "shop_vouchers": [],
        "shop_boosters": [],
        **kw,
    }


def test_heuristic_buys_the_better_joker_below_the_floor(trs):
    gs = _gs(6, shop_cards=[_card("Joker"), _card("Joker", x_mult=1.5)])
    agent = trs.HeuristicShopAgent()
    assert agent.act(gs, _mask(trs, 0, 1, 2, 4, 16)) == 1  # x_mult rates higher


def test_heuristic_skips_non_joker_cards(trs):
    gs = _gs(6, shop_cards=[_card("Tarot"), _card("Planet")])
    assert trs.HeuristicShopAgent().act(gs, _mask(trs, 0, 1, 15, 16)) == trs.SKIP


def test_heuristic_uses_planets_first(trs):
    gs = _gs(6, consumables=[_card("Tarot"), _card("Planet")], shop_cards=[_card("Joker")])
    assert trs.HeuristicShopAgent().act(gs, _mask(trs, 0, 13, 14, 16)) == 14


@pytest.mark.parametrize(
    ("dollars", "expected"),
    [(25, "SKIP"), (28, "pack"), (31, "voucher")],
)
def test_heuristic_keeps_the_interest_floor(trs, dollars, expected):
    gs = _gs(
        dollars,
        shop_vouchers=[_card("Voucher", cost=6)],
        shop_boosters=[_card("Booster", cost=4), _card("Booster", cost=3)],
    )
    a = trs.HeuristicShopAgent(interest_floor=25).act(gs, _mask(trs, 2, 3, 4, 15, 16))
    assert a == {"SKIP": trs.SKIP, "pack": 3, "voucher": trs.BUY_VOUCHER}[expected]


def test_heuristic_never_rerolls_or_sells(trs):
    gs = _gs(100)
    legal = _mask(trs, *trs.SELL_JOKER, *trs.SELL_CONS, trs.REROLL, trs.SKIP)
    assert trs.HeuristicShopAgent().act(gs, legal) == trs.SKIP


def test_smart_index_round_trips_every_shop_action(trs):
    agent = trs.SmartShopAgent()
    assert sorted(agent.index.values()) == list(range(trs.N_ACTIONS))
    for a in range(trs.N_ACTIONS):
        assert agent.index[trs._action_key(trs.to_factored(a))] == a


def test_smart_falls_back_to_skip(trs, monkeypatch):
    import jackdaw.cli.smart_agent as sa
    from jackdaw.engine import actions

    choice = actions.BuyCard(shop_index=2)  # a third shop slot (Overstock)
    monkeypatch.setattr(sa, "smart_agent", lambda gs, legal: choice)
    monkeypatch.setattr(actions, "get_legal_actions", lambda gs: [choice])
    agent = trs.SmartShopAgent()
    assert agent.act({}, _mask(trs, 0, 16)) == trs.SKIP
    choice = actions.BuyCard(shop_index=1)  # mappable, but ruled out by the mask
    assert agent.act({}, _mask(trs, 0, 16)) == trs.SKIP
    assert agent.fallbacks == {"BuyCard": 2}
    assert agent.act({}, _mask(trs, 1, 16)) == 1


@pytest.mark.slow
@pytest.mark.parametrize("shop", ["heuristic", "smart"])
def test_hand_written_shops_only_take_legal_actions(trs, tmp_path, monkeypatch, shop):
    args = [
        "--shop", shop, "--play", "smart", "--episodes", "6", "--log-every", "6",
        "--out-dir", str(tmp_path), "--run-name", shop,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train_random_shop.py", *args])
    trs.main()
    steps = [json.loads(line) for line in (tmp_path / shop / "steps.jsonl").open()]
    assert steps and all(s["action"] in s["legal"] for s in steps)
    config = json.loads((tmp_path / shop / "config.json").read_text())
    names = {"heuristic": "HeuristicShopAgent", "smart": "SmartShopAgent"}
    assert config["shop_agent"] == names[shop]
