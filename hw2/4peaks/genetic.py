from scipy.optimize import differential_evolution


def genetic(arr, fitness):
    bounds = [(0, 1) for i in range(len(arr))]
    res = differential_evolution(fitness, bounds)
    return res.x, res.fun