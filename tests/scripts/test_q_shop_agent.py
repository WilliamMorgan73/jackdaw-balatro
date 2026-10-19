"""The tabular Q-learning shop agent in scripts/train_random_shop.py."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

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


def _agent(trs, **kw):
    return trs.QLearningShopAgent(random.Random(0), random.Random(1), **kw)


S = np.array([1, 2, 0, 1, 0, 1, 0])
S2 = np.array([1, 1, 1, 0, 0, 1, 0])


@pytest.mark.parametrize("eps", [1.0, 0.0])
def test_act_only_picks_legal_actions(trs, eps):
    agent = _agent(trs, eps_start=eps, eps_end=eps)
    agent.q[tuple(S.tolist())] = np.arange(trs.N_ACTIONS, dtype=float)  # best = illegal 16
    legal = _mask(trs, 0, 3, 15)
    picks = {agent.act(S, legal) for _ in range(200)}
    assert picks <= {0, 3, 15}
    if eps == 0.0:
        assert picks == {15}  # highest legal value


def test_greedy_ties_are_broken_at_random(trs):
    agent = _agent(trs)
    picks = {agent.act(S, _mask(trs, 2, 7, 16), greedy=True) for _ in range(200)}
    assert picks == {2, 7, 16}


def test_update_matches_formula(trs):
    agent = _agent(trs, alpha=0.5, gamma=0.9)
    agent.q[tuple(S2.tolist())] = np.full(trs.N_ACTIONS, 2.0)
    agent.q[tuple(S2.tolist())][4] = 10.0  # illegal next action: must be ignored
    agent.observe(S, 3, 1.0, S2, False, _mask(trs, 0, 16))
    # 0 + 0.5 * (1 + 0.9 * max(2, 2) - 0)
    assert agent.values(S)[3] == pytest.approx(1.4)
    assert agent.visits[tuple(S.tolist())] == 1


def test_terminal_update_ignores_next_state(trs):
    agent = _agent(trs, alpha=0.5, gamma=0.9)
    agent.q[tuple(S2.tolist())] = np.full(trs.N_ACTIONS, 100.0)
    agent.observe(S, 3, 2.0, S2, True, np.zeros(trs.N_ACTIONS, dtype=bool))
    assert agent.values(S)[3] == pytest.approx(1.0)
    assert agent.episodes == 1


def test_epsilon_decays_linearly_then_holds(trs):
    agent = _agent(trs, eps_start=1.0, eps_end=0.1, eps_decay_episodes=10)
    zeros = np.zeros(trs.N_ACTIONS, dtype=bool)
    eps = []
    for _ in range(15):
        eps.append(agent.epsilon)
        agent.observe(S, 16, 0.0, S2, True, zeros)
    assert eps[0] == 1.0
    assert eps[5] == pytest.approx(0.55)
    assert eps[10:] == pytest.approx([0.1] * 5)


def test_values_does_not_grow_the_table(trs):
    agent = _agent(trs)
    agent.act(S, _mask(trs, 16), greedy=True)
    assert agent.values(S2).tolist() == [0.0] * trs.N_ACTIONS
    assert agent.q == {}


def test_tabular_state_ranges(trs):
    class J:
        center_key = "j_joker"

    gs = {"round_resets": {"ante": 11}, "dollars": -3, "jokers": [J()] * 7}
    s = trs.tabular_state(gs, _mask(trs, 0, 3, 16))
    assert s.tolist() == [8, 0, 5, 1, 0, 1, 0]
    gs = {"round_resets": {"ante": 2}, "dollars": 27, "jokers": []}
    assert trs.tabular_state(gs, _mask(trs, 4)).tolist() == [2, 5, 0, 0, 0, 0, 1]


@pytest.mark.slow
def test_training_run_is_reproducible(trs, tmp_path, monkeypatch):
    def run(name):
        args = [
            "--shop", "qlearn", "--play", "smart", "--episodes", "8",
            "--eval-every", "4", "--eval-episodes", "2",
            "--out-dir", str(tmp_path), "--run-name", name,
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", ["train_random_shop.py", *args])
        trs.main()
        return tmp_path / name

    a, b = run("a"), run("b")
    for f in ("steps.jsonl", "q_table.json", "eval/steps.jsonl"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    config = json.loads((a / "config.json").read_text())
    assert config["shop_agent"] == "QLearningShopAgent"
    assert config["agent_params"]["eps_decay_episodes"] == round(0.8 * 8)
