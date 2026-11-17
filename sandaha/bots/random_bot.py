import random


class RandomBot:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, game):
        return self.rng.choice(game.legal_actions())
