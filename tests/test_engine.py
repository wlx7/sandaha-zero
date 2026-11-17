import random
from sandaha.engine import Game, PLAY, OVER
from sandaha import moves as M


def play_random(seed):
    rng = random.Random(seed)
    g = Game(seed=seed, first_bidder=seed % 4)
    steps = 0
    while not g.over:
        acts = g.legal_actions()
        assert acts, (g.phase, g.to_act)
        a = rng.choice(acts)
        if g.phase == PLAY:
            err = g.check_play(g.to_act, a)
            assert err is None, (err, a, g.lead_shape)
        g.step(a)
        steps += 1
    return g


def test_random_games():
    tiers = {}
    for seed in range(300):
        g = play_random(seed)
        r = g.result
        assert sum(r["scores"]) == 0
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        if not r["surrender"]:
            # every card accounted for
            assert all(sum(h) == 0 for h in g.hands)
            assert len(g.tricks) >= 1
            total = sum(t["points"] for t in g.tricks) + M.points_of(g.buried)
            assert total == 200, total
    print(tiers)


if __name__ == "__main__":
    test_random_games()
