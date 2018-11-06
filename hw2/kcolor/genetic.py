from scipy.optimize import differential_evolution


def genetic(colors, k, fitness):
    bounds = [(1, k) for i in range(len(colors))]
    res = differential_evolution(fitness, bounds)
    return res.x, res.fun