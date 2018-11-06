
from simanneal import Annealer
import random

class KColorAnneal(Annealer):
    def __init__(self, colors, k, edges, fitness):
        self.colors = colors
        self.k = k
        self.edges = edges
        self.fitness = fitness
        super(KColorAnneal, self).__init__(colors)
    def move(self):
        randomi = random.randint(0, len(self.colors) - 1)
        randomc = random.randint(1, self.k)
        self.state[randomi] = randomc
    def energy(self):
        return self.fitness(self.state)
