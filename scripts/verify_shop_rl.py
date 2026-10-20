"""Feasibility checks for a shop-only DQN project built on Jackdaw.

Works through the "Things to Verify First" checklist for a ShopEnv wrapper
(17 shop actions, heuristic autoplay for every other phase, a deepcopy-based
play-phase oracle Phi(s)) and then runs a minimal prototype end to end.

Usage::

    uv run python scripts/verify_shop_rl.py [--episodes 20] [--play smart|strong]

``--play strong`` uses ``scripts/play_policy.py`` instead of ``smart_agent`` for
autoplay and inside Phi, so the oracle timing (item 4) is for that policy.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import Counter
from typing import Any

import numpy as np

from jackdaw.cli.smart_agent import smart_agent
from jackdaw.engine.actions import GamePhase
from jackdaw.env import BalatroEnvironment, DirectAdapter
from jackdaw.env.action_space import (
    ActionType as AT,
)
from jackdaw.env.action_space import (
    FactoredAction,
    engine_action_to_factored,
    get_consumable_target_info,
)

RESULTS: list[tuple[str, str, str]] = []
POLICY = smart_agent  # autoplay and Phi's play policy; --play strong swaps it
POLICY_NAME = "smart_agent"


def report(item: str, status: str, detail: str) -> None:
    RESULTS.append((item, status, detail))
    print(f"\n[{status}] {item}\n    " + detail.replace("\n", "\n    "))


def make_env(prefix: str = "VERIFY", **kw: Any) -> BalatroEnvironment:
    return BalatroEnvironment(adapter_factory=DirectAdapter, seed_prefix=prefix, **kw)


def phase_of(env: BalatroEnvironment) -> GamePhase:
    return env._adapter.raw_state["phase"]


def autoplay_action(env: BalatroEnvironment) -> FactoredAction:
    """The play policy's choice for the current state, as a FactoredAction."""
    from jackdaw.engine.actions import PickPackCard
    from jackdaw.env.action_space import get_action_mask

    gs = env._adapter.raw_state
    # get_legal_actions() offers pack picks that the executor rejects
    # (pack_pick_block_reason); the action mask doesn't, so filter by it.
    picks = get_action_mask(gs).entity_masks.get(AT.PickPackCard)
    legal = [
        a
        for a in env._adapter.get_legal_actions()
        if not isinstance(a, PickPackCard) or (picks is not None and picks[a.card_index])
    ]
    return engine_action_to_factored(POLICY(gs, legal), gs)


def advance_to(env: BalatroEnvironment, target: GamePhase, max_steps: int = 2000) -> tuple:
    """Step with the play policy until ``target`` phase (or game over)."""
    out = None
    for _ in range(max_steps):
        ph = phase_of(env)
        if ph == target or ph == GamePhase.GAME_OVER:
            break
        out = env.step(autoplay_action(env))
    return out


def leave_shop(env: BalatroEnvironment) -> None:
    env.step(FactoredAction(AT.NextRound))


# ---------------------------------------------------------------------------
# 1. raw_state keys
# ---------------------------------------------------------------------------


def check_raw_state() -> None:
    env = make_env()
    _, _, info = env.reset()
    gs = info["raw_state"]
    full_deck = gs["deck"] + gs["hand"] + gs["discard_pile"]
    c = full_deck[0]
    lines = [
        f"money: gs['dollars']={gs['dollars']}  "
        f"interest cap: gs['interest_cap']={gs['interest_cap']}"
        f"  (interest_amount={gs['interest_amount']})",
        f"joker slot limit: gs['joker_slots']={gs['joker_slots']}"
        f"  consumable slots: gs['consumable_slots']={gs['consumable_slots']}",
        "ante: gs['round_resets']['ante']  blind target: gs['blind'].chips  "
        "chips scored this blind: gs['chips']",
        f"deck: draw pile is gs['deck'] ({len(gs['deck'])}); full deck = deck + hand + discard_pile"
        f" ({len(full_deck)})",
        f"card fields: c.base.rank={c.base.rank.value!r} c.base.suit={c.base.suit.value!r} "
        f"c.base.id={c.base.id} (2..14)",
        f"  enhancement = c.center_key ({c.center_key!r} = none; m_glass, m_steel, ...)",
        f"  seal = c.seal ({c.seal!r}), edition = c.edition ({c.edition!r}), debuff = c.debuff",
    ]
    # Show what chips/blind look like mid-round
    advance_to(env, GamePhase.SELECTING_HAND)
    env.step(autoplay_action(env))
    gs = env._adapter.raw_state
    if gs["phase"] == GamePhase.SELECTING_HAND:
        lines.append(
            f"after 1 hand: gs['chips']={gs['chips']} / blind.chips={gs['blind'].chips} "
            f"(blind key {gs['blind'].key!r})"
        )
    lines.append("phase: gs['phase'] is a GamePhase enum -> no need to infer phase from the mask")
    report("1. raw_state keys", "OK", "\n".join(lines))


# ---------------------------------------------------------------------------
# 2. shop_item ordering
# ---------------------------------------------------------------------------


def check_shop_ordering() -> None:
    env = make_env()
    env.reset()
    advance_to(env, GamePhase.SHOP)
    gs = env._adapter.raw_state
    from jackdaw.env.observation import encode_observation

    obs = encode_observation(gs)
    n_c, n_v, n_b = len(gs["shop_cards"]), len(gs["shop_vouchers"]), len(gs["shop_boosters"])
    sets = [
        c.ability.get("set") for c in gs["shop_cards"] + gs["shop_vouchers"] + gs["shop_boosters"]
    ]
    col1 = obs.shop_cards[:, 1]
    lines = [
        f"shop_item rows = shop_cards({n_c}) + shop_vouchers({n_v}) + shop_boosters({n_b}),"
        " in that order",
        f"  sets per row: {sets}",
        f"  feature col 1 (card_set ordinal / 9): {np.round(col1, 3).tolist()}",
        "info['shop_splits'] already returns (n_cards, n_vouchers, n_boosters).",
        "IMPORTANT: entity_target is an index into the SUB-list, not the combined shop_item array:",
        "  BuyCard(i) -> shop_cards[i], RedeemVoucher(i) -> shop_vouchers[i],"
        " OpenBooster(i) -> shop_boosters[i]",
        "  so shop_slots() can be: card=range(n_c), voucher=range(n_v), pack=range(n_b) -- no"
        " card_set classifying needed.",
    ]
    # Verify OpenBooster(0) opens shop_boosters[0]
    gs["dollars"] = 100  # test-only: make everything affordable
    want = gs["shop_boosters"][0].center_key
    env.step(FactoredAction(AT.OpenBooster, entity_target=0))
    gs = env._adapter.raw_state
    ok = gs["phase"] == GamePhase.PACK_OPENING
    lines.append(
        f"verified: OpenBooster(0) opened {want!r} -> phase {gs['phase'].value}, "
        f"pack_type={gs.get('pack_type')!r}"
    )
    lines.append(
        "NOTE: the draft assumes '2 card slots / 2 pack slots / 1 voucher'. Counts change with"
        " vouchers (Overstock -> 3 cards) and the shop refills nothing after a buy (lists shrink),"
        " so index by position in the sub-list and mask out missing slots."
    )
    report("2. shop_item ordering / card_set", "OK" if ok else "FAIL", "\n".join(lines))


# ---------------------------------------------------------------------------
# 3 + 4. deepcopy + timing
# ---------------------------------------------------------------------------


def _fingerprint(gs: dict) -> tuple:
    return (
        gs["phase"].value,
        gs["dollars"],
        gs["chips"],
        gs["round_resets"]["ante"],
        tuple(c.card_key for c in gs["hand"]),
        tuple(c.center_key for c in gs["jokers"]),
        tuple(c.center_key for c in gs.get("shop_cards", [])),
    )


def _play_blind(env: BalatroEnvironment) -> tuple[int, int, bool]:
    """From the shop: leave, select blind, play greedily until blind ends.

    Returns (chips scored, blind target, beat_blind)."""
    leave_shop(env)
    advance_to(env, GamePhase.SELECTING_HAND)
    gs = env._adapter.raw_state
    target = gs["blind"].chips if gs.get("blind") else 0
    last_chips = 0
    while phase_of(env) == GamePhase.SELECTING_HAND:
        last_chips = env._adapter.raw_state["chips"]
        env.step(autoplay_action(env))
        last_chips = max(last_chips, env._adapter.raw_state["chips"])
    return last_chips, target, phase_of(env) != GamePhase.GAME_OVER


def check_deepcopy() -> None:
    env = make_env()
    env.reset()
    advance_to(env, GamePhase.SHOP)
    before = _fingerprint(env._adapter.raw_state)

    t0 = time.perf_counter()
    sim_a = copy.deepcopy(env)
    t_copy = time.perf_counter() - t0
    sim_b = copy.deepcopy(env)

    # Reroll uses RNG: identical copies must produce identical shops
    for s in (sim_a, sim_b):
        s._adapter.raw_state["dollars"] += 10
        s.step(FactoredAction(AT.Reroll))
    same_reroll = _fingerprint(sim_a._adapter.raw_state) == _fingerprint(sim_b._adapter.raw_state)

    t0 = time.perf_counter()
    res_a = _play_blind(sim_a)
    t_blind = time.perf_counter() - t0
    res_b = _play_blind(sim_b)
    same_blind = res_a == res_b and _fingerprint(sim_a._adapter.raw_state) == _fingerprint(
        sim_b._adapter.raw_state
    )
    untouched = _fingerprint(env._adapter.raw_state) == before

    # Does the original, played forward the same way, match the copy? (RNG really copied)
    sim_c = copy.deepcopy(env)
    res_c = _play_blind(sim_c)
    res_orig = _play_blind(env)
    orig_matches = res_c == res_orig

    ok = same_reroll and same_blind and untouched and orig_matches
    detail = (
        f"copy.deepcopy(BalatroEnvironment) works: {True}\n"
        f"two copies reroll identically (RNG state copied): {same_reroll}\n"
        f"two copies play the next blind identically: {same_blind}  result={res_a}\n"
        f"original env untouched by simulating on copies: {untouched}\n"
        f"copy's blind == original's blind when both played: {orig_matches}\n"
        f"timing: deepcopy {t_copy * 1000:.1f} ms, simulate one blind {t_blind * 1000:.1f} ms"
    )
    report("3. deepcopy(env) incl. RNG", "OK" if ok else "FAIL", detail)

    # Item 4: timing over many shop states
    copy_t, blind_t = [], []
    for ep in range(5):
        e = make_env("TIMING")
        e.reset(seed=f"TIMING_{ep}")
        for _ in range(4):
            advance_to(e, GamePhase.SHOP)
            if phase_of(e) != GamePhase.SHOP:
                break
            t0 = time.perf_counter()
            s = copy.deepcopy(e)
            copy_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _play_blind(s)
            blind_t.append(time.perf_counter() - t0)
            leave_shop(e)
    phi = np.array(copy_t) + np.array(blind_t)
    detail = (
        f"n={len(phi)} shop states (antes 1-2)\n"
        f"deepcopy: mean {np.mean(copy_t) * 1000:.1f} ms, max {np.max(copy_t) * 1000:.1f} ms\n"
        f"simulate blind ({POLICY_NAME} policy): mean {np.mean(blind_t) * 1000:.1f} ms,"
        f" max {np.max(blind_t) * 1000:.1f} ms\n"
        f"=> one Phi(s) call ~ {np.mean(phi) * 1000:.0f} ms  "
        f"(~{1 / np.mean(phi):.0f} oracle calls/s"
        " per core)\n"
        "GreedyOracleAgent calls Phi once per legal action (often 8-15) -> ~1 s per decision;"
        " fine for evaluation, too slow to train against without caching/multiprocessing."
    )
    report("4. Oracle cost (deepcopy + simulate blind)", "INFO", detail)


# ---------------------------------------------------------------------------
# 5. hand-scoring function
# ---------------------------------------------------------------------------


def check_scoring() -> None:
    from jackdaw.cli.smart_agent import _pick_best_hand_scored
    from jackdaw.engine.scoring import score_hand

    env = make_env()
    env.reset()
    advance_to(env, GamePhase.SELECTING_HAND)
    gs = env._adapter.raw_state
    idx, htype, est = _pick_best_hand_scored(gs["hand"], gs["jokers"], gs["hand_levels"])

    # Exact engine score on a throwaway copy (score_hand consumes RNG)
    sim = copy.deepcopy(gs)
    played = [sim["hand"][i] for i in idx]
    held = [c for i, c in enumerate(sim["hand"]) if i not in set(idx)]
    res = score_hand(
        played,
        held,
        sim["jokers"],
        sim["hand_levels"],
        sim["blind"],
        sim["rng"],
        game_state=sim,
        back_key=sim["selected_back_key"],
        blind_chips=sim["blind"].chips,
    )
    detail = (
        "Two options exist:\n"
        f"  a) jackdaw.cli.smart_agent._pick_best_hand_scored(hand, jokers, hand_levels) -> "
        f"(indices, type, est)\n     here: {idx} {htype!r} est={est}  (fast estimate, approximates"
        " common joker effects, no RNG)\n"
        "  b) jackdaw.engine.scoring.score_hand(played, held, jokers, hand_levels, blind, rng, ...)"
        f" -> exact\n     here: total={getattr(res, 'total', res)}  (full joker pipeline; ADVANCES"
        " RNG, so call it on a deepcopy)\n"
        "smart_agent() itself is a full greedy play/discard policy you can reuse for autoplay."
    )
    report("5. Hand-scoring function", "OK", detail)


# ---------------------------------------------------------------------------
# 6. pack type, 9. NextRound during packs
# ---------------------------------------------------------------------------


def check_packs(episodes: int) -> None:
    kinds: Counter[str] = Counter()
    nextround_in_pack = 0
    pack_states = 0
    picks_masked_all = 0
    for ep in range(episodes):
        env = make_env("PACKS")
        env.reset(seed=f"PACKS_{ep}")
        for _ in range(6):
            advance_to(env, GamePhase.SHOP)
            gs = env._adapter.raw_state
            if gs["phase"] != GamePhase.SHOP:
                break
            if gs["shop_boosters"]:
                gs["dollars"] = max(gs["dollars"], 20)
                _, _, _, mask, info = env.step(FactoredAction(AT.OpenBooster, entity_target=0))
                gs = info["raw_state"]
                if gs["phase"] == GamePhase.PACK_OPENING:
                    pack_states += 1
                    kinds[gs.get("pack_type")] += 1
                    nextround_in_pack += bool(mask.type_mask[AT.NextRound])
                    picks_masked_all += not mask.type_mask[AT.PickPackCard]
                    while phase_of(env) == GamePhase.PACK_OPENING:
                        env.step(FactoredAction(AT.SkipPack))
            if phase_of(env) == GamePhase.SHOP:
                leave_shop(env)
    report(
        "6. Which pack type is open",
        "OK",
        f"gs['pack_type'] holds the kind while phase == PACK_OPENING. Seen: {dict(kinds)}\n"
        "Pack cards: gs['pack_cards'] (Card list), picks left: gs['pack_choices_remaining'].\n"
        "Tarot/Spectral packs deal a hand (gs['pack_hand']); PickPackCard without card_target"
        " auto-fills default targets.",
    )
    report(
        "9. NextRound legal while a pack is open?",
        "OK" if nextround_in_pack == 0 else "FAIL",
        f"{pack_states} pack states checked: NextRound legal in {nextround_in_pack}.\n"
        "Phase is PACK_OPENING (not SHOP) while a pack is open; after the last pick/skip it"
        " returns to SHOP (if opened from the shop). Just switch on gs['phase'].\n"
        f"(packs where every pick was masked -> only SkipPack legal: {picks_masked_all})",
    )


# ---------------------------------------------------------------------------
# 7. consumable targeting
# ---------------------------------------------------------------------------


def check_consumable_targets() -> None:
    from jackdaw.engine.card_factory import create_consumable

    rows = []
    for key in [
        "c_magician",
        "c_strength",
        "c_death",
        "c_hermit",
        "c_judgement",
        "c_jupiter",
        "c_aura",
    ]:
        rows.append(
            f"{key:12s} -> (min, max, needs_targets) = "
            f"{get_consumable_target_info(create_consumable(key))}"
        )

    # In the shop there is no hand, so targeting consumables should be masked out
    env = make_env()
    env.reset()
    advance_to(env, GamePhase.SHOP)
    gs = env._adapter.raw_state
    gs["consumables"] = [create_consumable("c_magician"), create_consumable("c_hermit")]
    from jackdaw.env.action_space import get_action_mask

    m = get_action_mask(gs).entity_masks.get(AT.UseConsumable)
    report(
        "7. Detect 'needs a card target' for UseConsumable",
        "OK" if m is not None and not m[0] and m[1] else "FAIL",
        "jackdaw.env.action_space.get_consumable_target_info(card) -> (min, max, needs_targets)\n"
        + "\n".join(rows)
        + "\nShop mask for [Magician, Hermit] held: "
        f"UseConsumable = {None if m is None else m.tolist()}"
        "\n=> in the SHOP the engine already masks targeting consumables (no hand to target),"
        " so the extra needs_card_target() check in shop_mask() is redundant but harmless.",
    )


# ---------------------------------------------------------------------------
# 8. max jokers
# ---------------------------------------------------------------------------


def check_joker_slots() -> None:
    from jackdaw.engine.run_init import initialize_run
    from jackdaw.env.balatro_spec import balatro_game_spec

    black = initialize_run("b_black", 1, "X")["joker_slots"]
    joker_cap = next(e.max_count for e in balatro_game_spec().entity_types if e.name == "joker")
    lines = [
        "Base joker_slots = 5. Ways to hold more than 5 jokers:",
        f"  Black Deck (b_black): joker_slots={black}",
        "  Antimatter voucher (v_antimatter): +1 slot",
        "  Negative-edition jokers: +1 slot each (unbounded in principle)",
        "Engine masks (SellJoker etc.) are sized to len(jokers), no cap at 5.",
        "Jackdaw's GameSpec/gym wrapper pads joker entities to "
        f"max_count={joker_cap} (excess truncated) -- only matters if you use"
        " BalatroGymnasiumEnv.",
        "=> 6 sell-joker actions covers Red deck + one Negative joker; Black deck, Antimatter"
        " or several Negatives can exceed it (relevant to SRQ4). Either keep 6 and"
        " accept masking, or go to 7-8.",
    ]
    report("8. Joker slots > 5?", "INFO", "\n".join(lines))


# ---------------------------------------------------------------------------
# End-to-end prototype
# ---------------------------------------------------------------------------


def shop_actions(gs: dict, mask) -> list[FactoredAction]:
    """All legal shop actions in the draft's 17-action space (approximately)."""
    em, tm = mask.entity_masks, mask.type_mask
    out = []
    for at in (
        AT.BuyCard,
        AT.OpenBooster,
        AT.RedeemVoucher,
        AT.SellJoker,
        AT.SellConsumable,
        AT.UseConsumable,
    ):
        if tm[at]:
            out += [FactoredAction(at, entity_target=int(i)) for i in np.nonzero(em[at])[0]]
    if tm[AT.Reroll]:
        out.append(FactoredAction(AT.Reroll))
    out.append(FactoredAction(AT.NextRound))
    return out


def run_prototype(episodes: int) -> None:
    antes, wins, shop_steps, errors = [], [], 0, []
    t0 = time.perf_counter()
    for ep in range(episodes):
        env = make_env("EVAL")
        _, mask, info = env.reset(seed=f"EVAL_{ep}")
        try:
            for _ in range(5000):
                gs = env._adapter.raw_state
                if gs["phase"] == GamePhase.GAME_OVER:
                    break
                if gs["phase"] == GamePhase.SHOP:
                    a = random.choice(shop_actions(gs, info["action_mask"]))
                    shop_steps += 1
                else:
                    a = autoplay_action(env)
                _, term, trunc, mask, info = env.step(a)
                if term or trunc:
                    break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ep {ep}: {type(exc).__name__}: {exc}")
        antes.append(env.episode_ante)
        wins.append(env.episode_won)
    dt = time.perf_counter() - t0
    detail = (
        f"{episodes} EVAL-seed runs, random shop agent + {POLICY_NAME} autoplay\n"
        f"mean ante {np.mean(antes):.2f} (min {min(antes)}, max {max(antes)}), wins {sum(wins)}\n"
        f"{shop_steps} shop decisions, {dt:.1f}s total ({dt / episodes:.2f}s/run without oracle)\n"
        f"errors: {errors or 'none'}"
    )
    report("E2E. Full runs through a shop-only loop", "OK" if not errors else "FAIL", detail)

    # Determinism of seed_prefix
    def first_shop(prefix: str) -> tuple:
        e = make_env(prefix)
        e.reset()
        advance_to(e, GamePhase.SHOP)
        return _fingerprint(e._adapter.raw_state)

    same = first_shop("EVAL") == first_shop("EVAL")
    diff = first_shop("EVAL") != first_shop("TRAIN")
    report(
        "E2E. seed_prefix determinism",
        "OK" if same and diff else "FAIL",
        f"same prefix -> identical first shop: {same}; EVAL vs TRAIN differ: {diff}\n"
        "Seeds are f'{prefix}_{n}' where n counts resets of THAT env object: reuse one env to\n"
        "get distinct runs (a fresh env per episode replays seed _0 every time).\n"
        "NOTE: with several back_keys/stakes, BalatroEnvironment picks them with the global"
        " `random` module -> call random.seed(...) for reproducible SRQ4 sweeps, or construct"
        " one env per (deck, stake).",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument(
        "--play",
        choices=["smart", "strong"],
        default="smart",
        help="play policy for autoplay and Phi: smart_agent, or scripts/play_policy.py",
    )
    args = ap.parse_args()
    global POLICY, POLICY_NAME
    if args.play == "strong":
        from play_policy import strong_agent  # scripts/play_policy.py, next to this file

        POLICY, POLICY_NAME = strong_agent, "strong_agent"
    random.seed(0)

    check_raw_state()
    check_shop_ordering()
    check_deepcopy()
    check_scoring()
    check_packs(args.episodes)
    check_consumable_targets()
    check_joker_slots()
    run_prototype(args.episodes)

    print("\n" + "=" * 70)
    for item, status, _ in RESULTS:
        print(f"[{status:4s}] {item}")


if __name__ == "__main__":
    main()
