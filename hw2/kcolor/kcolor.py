import copy
from anneal import KColorAnneal
from hill_climb import climb
from genetic import genetic


k = 5

colors = [1 for i in range(100)]
edges = set([(1, 2), (2, 1), (3, 5), (5, 3), (3, 6), (6, 3), (3, 9), (9, 3), (10, 3), (3, 10)])

def fitness(colors):
    cost = 0
    for i in range(len(colors)):
        for j in range(len(colors)):
            if (i, j) in edges and colors[i] == colors[j]:
                cost += 1
    return cost

def convert(arr):
    arr2 = []
    for i in arr:
        arr2.append(int(round(i)))
    return arr2

def fitness_gen(arr):
    return fitness(convert(arr))

print("Simulated annealing:")
edges2 = copy.deepcopy(edges)
colors2 = copy.deepcopy(colors)
anneal = KColorAnneal(colors2, k, edges2, fitness)
opt_colors, opt_fit = anneal.anneal()
print(opt_colors)
print(opt_fit)

print("Hill climbing:")
edges2 = copy.deepcopy(edges)
colors2 = copy.deepcopy(colors)
opt_colors, opt_fit = climb(colors2, k, edges2, fitness)
print(opt_colors)
print(opt_fit)


print("Genetic algo:")
edges2 = copy.deepcopy(edges)
colors2 = copy.deepcopy(colors)
opt_colors, opt_fit = genetic(colors2, k, fitness_gen)
print(convert(opt_colors))
print(opt_fit)

