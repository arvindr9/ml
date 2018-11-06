import copy

colors = [1 for i in range(100)]
edges = set((1, 2), (2, 1), (3, 5), (5, 3), (3, 6), (6, 3), (3, 9), (9, 3), (10, 3), (3, 10))

def fitness(colors, edges, k):
    cost = 0
    for i in range(len(colors)):
        for j in range(len(colors)):
            if (i, j) in edges and colors[i] == colors[j]:
                cost += 1
    return cost

edges2 = copy.deepcopy(edges)
colors2 = copy.deepcopy(colors)
anneal(colors2, k, edges2, fitness)

edges2 = copy.deepcopy(edges)
colors2 = copy.deepcopy(colors)
genetic(colors2, k, edges2, fitness)

edges2 = copy.deepcopy(edges)
colors2 = copy.deepcopy(colors)
climb(colors2, k, edges2, fitness)