"""Head-to-head evaluation with duplicate deals.

Team A sits at seats (0, 2), team B at (1, 3); every deal is replayed with the
teams swapped so luck of the cards cancels out.  Because the game is zero-sum,
team A "wins" a game when the sum of its two seats' scores is positive.
"""
from __future__ import annotations

import random

from .cards import full_deck
from .engine import HAND_SIZE, Game
from .rules import Rules


def random_deal(rng):
    deck = full_deck()
    rng.shuffle(deck)
    return [deck[i * HAND_SIZE:(i + 1) * HAND_SIZE] for i in range(4)], deck[4 * HAND_SIZE:]


def play_game(agents, rules=None, hands=None, bottom=None, first_bidder=0, seed=None):
    g = Game(rules, seed=seed, first_bidder=first_bidder, hands=hands, bottom=bottom)
    while not g.over:
        g.step(agents[g.to_act].act(g))
    return g


def duel(agent_a, agent_b, n_deals=100, rules=None, seed=0, progress=None):
    rules = rules or Rules()
    rng = random.Random(seed)
    stats = {"games": 0, "a_score": 0.0, "a_wins": 0.0, "a_banker_games": 0, "a_banker_wins": 0,
             "b_banker_games": 0, "b_banker_wins": 0, "a_xian_games": 0, "a_xian_wins": 0}
    for d in range(n_deals):
        hands, bottom = random_deal(rng)
        fb = rng.randrange(4)
        for swap in (0, 1):
            seats = [agent_a, agent_b, agent_a, agent_b] if swap == 0 else [agent_b, agent_a, agent_b, agent_a]
            a_seats = (0, 2) if swap == 0 else (1, 3)
            g = play_game(seats, rules, hands, bottom, fb, seed=rng.randrange(1 << 30))
            sc = g.result["scores"]
            a = sum(sc[s] for s in a_seats)
            stats["games"] += 1
            stats["a_score"] += a
            stats["a_wins"] += 1.0 if a > 0 else (0.5 if a == 0 else 0.0)
            banker = g.result["banker"]
            banker_won = not g.result["xian_win"]
            if banker in a_seats:
                stats["a_banker_games"] += 1
                stats["a_banker_wins"] += banker_won
            else:
                stats["b_banker_games"] += 1
                stats["b_banker_wins"] += banker_won
                stats["a_xian_games"] += 1
                stats["a_xian_wins"] += not banker_won
        if progress:
            progress(d + 1, n_deals)
    n = max(1, stats["games"])
    return {
        "games": stats["games"],
        "win_rate": stats["a_wins"] / n,
        "avg_score": stats["a_score"] / n,
        "banker_rate": stats["a_banker_games"] / n,
        "banker_win_rate": stats["a_banker_wins"] / max(1, stats["a_banker_games"]),
        "opp_banker_win_rate": stats["b_banker_wins"] / max(1, stats["b_banker_games"]),
    }
