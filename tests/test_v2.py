"""Unit tests for v2 engine deductions, tactical features, Dueling FiLM-ResNet, and SearchAgent."""
import numpy as np
import torch

from sandaha import moves as M
from sandaha.ai import features as F
from sandaha.ai.model import PRESETS, Model, param_count, save_params
from sandaha.ai.search import SearchAgent, sample_world
from sandaha.cards import CODE_TO_TYPE as C, NO_TRUMP, TRUMP_CLASS, trump_context
from sandaha.engine import PASS, PLAY, Game
from sandaha.rules import Rules
from sandaha.train.torch_model import SanDaHaNet


def t(*codes):
    return tuple(sorted(C[c] for c in codes))


def hand_of(*codes):
    h = [0] * 46
    for c in codes:
        h[C[c]] += 1
    return h


def test_passed_unforced_tracking():
    g = Game(Rules(force_enabled=True), seed=10, first_bidder=0)
    # Force hands so seat 1 has 0 changzhu and passes at 75
    g.hands = [hand_of(*(["SA", "SK", "SQ", "SJ", "S10", "S9", "S8", "S6", "S5", "HA", "HK"] + ["C5"] * 10)) for _ in range(4)]
    g.step(75)          # seat 0 bids 75 (> force_bid=70)
    g.step(PASS)        # seat 1 passes unforced -> provably < 7 changzhu
    assert g.passed_unforced[1] is True
    g.step(70)          # seat 2 bids 70 (<= force_bid=70)
    g.step(PASS)        # seat 3 passes at 70 -> NOT an unforced pass above force_bid!
    assert g.passed_unforced[3] is False


def test_pair_void_tracking_and_trump_throw_with_buried_trumps():
    g = Game(Rules(), seed=1, first_bidder=0)
    g.phase = PLAY
    g.banker = 0
    g.contract = 60
    g.trump = 1  # Hearts
    g.ctx = trump_context(1)
    # Buried contains two big trumps (BJ, BJ)!
    g.buried = [C["BJ"], C["BJ"], C["C5"], C["C5"], C["D5"], C["D5"], C["S5"], C["S5"]]
    # Seat 0 has SA SA + H9 H8; Seat 1 has SK SQ (2 spades, no pair) + C6 C6; Seat 2 has SJ S9 + D6 D6; Seat 3 has C8 C9 + D8 D9
    g.hands = [
        hand_of("SA", "SA", "H9", "H8"),
        hand_of("SK", "SQ", "C6", "C6"),
        hand_of("SJ", "S9", "D6", "D6"),
        hand_of("C8", "C9", "D8", "D9"),
    ]
    g.played_total = [2 - sum(g.hands[u][k] for u in range(4)) for k in range(46)]
    for b in g.buried:
        g.played_total[b] -= 1
    g.leader = g.to_act = 0

    # Trick 1: Seat 0 leads SA SA
    g.step(t("SA", "SA"))
    g.step(t("SK", "SQ"))  # Seat 1 followed with two singles -> pair_void in Spades (cls=0)!
    g.step(t("SJ", "S9"))  # Seat 2 followed with two singles -> pair_void in Spades!
    g.step(t("C8", "C9"))  # Seat 3 off-suit -> void & pair_void in Spades!
    assert g.pair_void[1][0] is True
    assert g.pair_void[2][0] is True
    assert g.void[3][0] is True and g.pair_void[3][0] is True

    # Suppose all opponents are now known void in TRUMP_CLASS, even if Seat 1 didn't know BJ BJ were buried:
    for o in (0, 2, 3):
        g.void[o][TRUMP_CLASS] = True
        g.pair_void[o][TRUMP_CLASS] = True
    g.hands[1] = hand_of("HA", "HK")
    g.to_act = 1
    g.cur = []
    # Seat 1 does NOT know g.buried (which holds BJ, BJ), yet all opponents are void in trump -> throw HA+HK MUST be offered!
    plays = g.legal_plays(1)
    assert t("HA", "HK") in plays


def test_extreme_v2_feature_bounds():
    # Burying 80 points (8 Ks/10s) and holding 21 trumps must NEVER exceed 1.0 in v2!
    g = Game(Rules(), seed=2, first_bidder=0)
    g.banker = 0
    g.contract = 55
    g.trump = 0
    g.ctx = trump_context(0)
    g.phase = PLAY
    g.buried = [C["HK"], C["HK"], C["H10"], C["H10"], C["CK"], C["CK"], C["C10"], C["C10"]]  # 80 points!
    g.hands[0] = hand_of(
        "BJ", "BJ", "LJ", "LJ", "S7", "S7", "H7", "H7", "C7", "C7", "D7", "D7",
        "S2", "S2", "H2", "H2", "SA", "SA", "SK", "SK", "SQ",
    )
    g.leader = g.to_act = 0
    acts = g.legal_actions()
    ph, st, A = F.encode(g, 0, acts, feat_ver=2)
    assert st.min() >= 0.0 and st.max() <= 1.0
    assert A.min() >= 0.0 and A.max() <= 1.0


def test_dueling_net_roundtrip_and_search(tmp_path):
    arch = PRESETS["tiny"]
    net = SanDaHaNet(arch, arch_type="dueling_film_v2", feat_ver=2)
    assert sum(p.numel() for p in net.parameters()) == param_count(arch, "dueling_film_v2", 2)
    p_path = str(tmp_path / "test_v2.npz")
    save_params(p_path, net.to_numpy(), {"name": "t", "arch": arch, "arch_type": "dueling_film_v2", "feat_ver": 2})
    m = Model.load(p_path)
    assert m.arch_type == "dueling_film_v2" and m.feat_ver == 2

    sa = SearchAgent(m, samples=4, topk=2, gap=5.0, seed=7, max_rollouts=12)
    g = Game(seed=99)
    while not g.over:
        g.step(sa.act(g))
    assert sum(g.result["scores"]) == 0
