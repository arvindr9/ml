import copy
from anneal import KColorAnneal
from hill_climb import climb
from genetic import genetic


k = 20
l = 50 #number of nodes: update this between runs

colors = [1 for i in range(l)]
edges = set() #this is a clique
for i in range(l):
    for j in range(l):
        if i != j:
            edges.add((i, j))
# edges = set([(1, 2), (2, 1), (3, 5),
# (5, 3), (3, 6), (6, 3), (3, 9), (9, 3),
# (10, 3), (3, 10), (25, 10), (25, 2), (25, 3),
# (25, 4), (25, 5), (25, 6), (10, 25), (2, 25),
# (3, 25), (4, 25), (5, 25), (35, 36), (36, 35),
# (34, 36), (36, 34), (33, 35), (35, 33), (7, 35),
# (35, 7), (6, 25), (7, 36), (36, 7), (8, 36), (36, 8),
# (9, 36), (36, 9), (10, 36), (36, 10), (11, 36),
# (36, 11), (12, 36), (36, 12), (13, 36), (36, 13),
# (4, 8), (8, 4), ()])

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
anneal.steps = 5000
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

