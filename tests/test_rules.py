"""Unit tests for the rule details that matter most."""
from sandaha import moves as M
from sandaha.cards import BJ, CODE_TO_TYPE as C, NO_TRUMP, SJ, trump_context
from sandaha.engine import BID, PASS, PLAY, SURRENDER, TRUMP, Game
from sandaha.rules import Rules

H = 1  # hearts
CTX = trump_context(H)


def hand_of(*codes):
    h = [0] * 46
    for c in codes:
        h[C[c]] += 1
    return h


def t(*codes):
    return tuple(sorted(C[c] for c in codes))


# ----------------------------------------------------------- ordering ---

def test_trump_order():
    order = ["BJ", "LJ", "H7", "S7", "H2", "S2", "HA", "HK", "HQ", "HJ", "H10", "H9", "H8", "H6", "H5"]
    pos = [CTX.pos[C[c]] for c in order]
    assert pos == sorted(pos) and len(set(pos)) == len(pos)
    assert CTX.pos[C["S7"]] == CTX.pos[C["D7"]] == CTX.pos[C["C7"]]  # 副7 equal
    assert all(CTX.is_trump[C[c]] for c in order + ["C2", "D7"])
    assert not CTX.is_trump[C["SA"]]


def test_no_trump_order():
    ctx = trump_context(NO_TRUMP)
    assert ctx.is_trump[C["H7"]] and ctx.is_trump[C["S2"]] and not ctx.is_trump[C["HA"]]
    assert ctx.pos[BJ] < ctx.pos[SJ] < ctx.pos[C["H7"]] < ctx.pos[C["H2"]]


# ------------------------------------------------------------ shapes ---

def test_side_tractor_skips_seven():
    sh = M.analyze(t("S8", "S8", "S6", "S6"), CTX)
    assert sh.kind == M.TRACTOR and sh.pairs == 2


def test_trump_tractors():
    for cards in [("BJ", "BJ", "LJ", "LJ"), ("H7", "H7", "S7", "S7"), ("S7", "S7", "H2", "H2"),
                  ("H2", "H2", "C2", "C2"), ("D2", "D2", "HA", "HA"), ("HA", "HA", "HK", "HK")]:
        assert M.analyze(t(*cards), CTX).kind == M.TRACTOR, cards


def test_not_tractors():
    # two different 副7 pairs have equal rank -> not consecutive
    assert M.analyze(t("S7", "S7", "C7", "C7"), CTX).kind == M.THROW
    # 小王 and 副7 are not adjacent (主7 between)
    assert M.analyze(t("LJ", "LJ", "S7", "S7"), CTX).kind == M.THROW
    # different suits
    assert M.analyze(t("SA", "SA", "SK"), CTX).kind == M.THROW


def test_side_throw_forbidden():
    h = hand_of("SA", "SK")
    assert M.check_lead(h, t("SA", "SK"), CTX) is not None
    h = hand_of("BJ", "LJ")
    assert M.check_lead(h, t("BJ", "LJ"), CTX) is None  # trump throw allowed


# ---------------------------------------------------------- following ---

def test_must_follow_pair():
    lead = M.analyze(t("SQ", "SQ"), CTX)
    h = hand_of("S5", "S5", "S9", "HA")
    assert M.check_follow(h, lead, t("S9", "S5"), CTX) is not None
    assert M.check_follow(h, lead, t("S5", "S5"), CTX) is None
    assert set(M.follow_moves(h, lead, CTX)) == {t("S5", "S5")}


def test_must_follow_suit_then_fill():
    lead = M.analyze(t("SQ", "SQ"), CTX)
    h = hand_of("S9", "HA", "D5")
    assert M.check_follow(h, lead, t("HA", "D5"), CTX) is not None
    assert M.check_follow(h, lead, t("S9", "D5"), CTX) is None
    assert all(C["S9"] in mv for mv in M.follow_moves(h, lead, CTX))


def test_tractor_follow_with_pairs():
    lead = M.analyze(t("SA", "SA", "SK", "SK"), CTX)
    h = hand_of("S9", "S9", "S5", "S5", "S6", "HA")  # pairs 99 and 55 are not adjacent (8, 6 between)
    mv = M.follow_moves(h, lead, CTX)
    assert mv and all(m.count(C["S9"]) == 2 and m.count(C["S5"]) == 2 for m in mv)


def test_trick_trumping():
    plays = [(0, t("SA")), (1, t("S5")), (2, t("H5")), (3, t("SK"))]
    assert M.trick_winner(plays, CTX) == 2
    plays = [(0, t("SA")), (1, t("H5")), (2, t("H6")), (3, t("BJ"))]
    assert M.trick_winner(plays, CTX) == 3
    # equal 副7: first played wins
    plays = [(0, t("S7")), (1, t("C7")), (2, t("D7")), (3, t("H5"))]
    assert M.trick_winner(plays, CTX) == 0
    # pair lead cannot be beaten by two trump singles
    plays = [(0, t("SQ", "SQ")), (1, t("H5", "H6")), (2, t("S5", "S6")), (3, t("H9", "H9"))]
    assert M.trick_winner(plays, CTX) == 3


def test_throw_failure_forces_component():
    hands = [hand_of("BJ", "BJ", "H9"), hand_of("HJ"), hand_of("C5"), hand_of("D5")]
    ok, forced = M.throw_result(t("BJ", "BJ", "H9"), 0, hands, CTX)
    assert not ok and forced == t("H9")
    hands[1] = hand_of("C5")
    ok, forced = M.throw_result(t("BJ", "BJ", "H9"), 0, hands, CTX)
    assert ok


# ------------------------------------------------------------ bidding ---

def _deal_with(changzhu_seat):
    hands = [[] for _ in range(4)]
    from sandaha.cards import full_deck
    deck = full_deck()
    cz = [x for x in deck if x >= 44 or x % 11 in (0, 3)]
    others = [x for x in deck if not (x >= 44 or x % 11 in (0, 3))]
    hands[changzhu_seat] = cz[:8] + others[:13]
    others = others[13:] + cz[8:]
    for s in range(4):
        if s != changzhu_seat:
            hands[s], others = others[:21], others[21:]
    return hands, others


def test_forced_bid():
    hands, bottom = _deal_with(1)
    g = Game(Rules(), hands=hands, bottom=bottom, first_bidder=0)
    g.step(80)
    assert PASS not in g.legal_bids()  # seat 1 holds 8 常主
    g.step(75)
    g.step(PASS)
    g.step(PASS)
    g.step(70)                         # seat 0 undercuts to 70
    assert PASS in g.legal_bids()      # bid already <= 70: seat 1 may stop now
    g.step(PASS)
    assert g.banker == 0 and g.contract == 70 and g.phase == TRUMP


def test_all_pass_first_bidder_80():
    g = Game(Rules(force_enabled=False), seed=3, first_bidder=2)
    for _ in range(4):
        g.step(PASS)
    assert g.banker == 2 and g.contract == 80


# --------------------------------------------------------- settlement ---

def _settle(points, contract, **kw):
    g = Game(Rules(**kw), seed=1)
    g.banker, g.contract, g.xian_points = 0, contract, points
    g._settle()
    return g.result


def test_settlement_table():
    # Default rules have shuang_jin_dan_chu=True (banker wins doubled)
    assert _settle(0, 60)["tier"] == "大光" and _settle(0, 60)["scores"] == [18, -6, -6, -6]
    assert _settle(30, 60)["tier"] == "小光" and _settle(30, 60)["scores"] == [12, -4, -4, -4]
    assert _settle(35, 60)["tier"] == "保庄" and _settle(35, 60)["scores"] == [6, -2, -2, -2]
    assert _settle(60, 60)["tier"] == "下庄" and _settle(60, 60)["scores"] == [-3, 1, 1, 1]
    assert _settle(100, 60)["tier"] == "小倒" and _settle(100, 60)["scores"][1] == 2
    assert _settle(130, 60)["tier"] == "大倒" and _settle(130, 60)["scores"][0] == -9
    # Single-in single-out (shuang_jin_dan_chu=False)
    assert _settle(0, 60, shuang_jin_dan_chu=False)["scores"] == [9, -3, -3, -3]
    assert _settle(35, 60, shuang_jin_dan_chu=False)["scores"][0] == 3


def test_surrender():
    g = Game(Rules(force_enabled=False), seed=5, first_bidder=0)
    g.step(60)
    for _ in range(3):
        g.step(PASS)
    g.step(SURRENDER)
    assert g.over and g.result["tier"].startswith("投降") and g.result["scores"][0] == -3
    g = Game(Rules(force_enabled=False), seed=5, first_bidder=0)
    g.step(40)
    for _ in range(3):
        g.step(PASS)
    g.step(SURRENDER)
    assert g.result["scores"][0] == -6


def test_koudi_multiplier():
    g = Game(Rules(), seed=1)
    g.banker, g.contract, g.trump, g.ctx = 0, 60, H, CTX
    g.buried = [C["SK"], C["S10"], C["S5"]] + [C["S6"]] * 5  # 25 points
    lead = M.analyze(t("H9", "H9"), CTX)
    g._koudi(1, lead, t("H9", "H9"))
    assert g.koudi["gained"] == 50
    g.xian_points = 0
    g._koudi(1, M.analyze(t("SA"), CTX), t("SA"))  # side card cannot 扣底
    assert g.koudi["gained"] == 0
