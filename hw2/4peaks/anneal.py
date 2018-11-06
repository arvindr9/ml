
from simanneal import Annealer
import random

class FourPeaksAnneal(Annealer):
    def __init__(self, arr, fitness):
        self.arr = arr
        self.fitness = fitness
        super(FourPeaksAnneal, self).__init__(arr)
    def move(self):
        randomi = random.randint(0, len(self.arr) - 1)
        randomc = random.randint(0, 1)
        self.state[randomi] = randomc
    def energy(self):
        return self.fitness(self.state)
