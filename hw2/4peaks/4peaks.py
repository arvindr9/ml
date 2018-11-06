import copy

t = 4

arr = [0 for i in range(100)]

def fitness(arr, t):
    res = 0
    n1 = 0
    n2 = 0
    for i in arr:
        if i != 0:
            break
        n1 += 1
    for i in list(reversed(arr)):
        if i != 1:
            break
        n2 += 1
    if n1 > t and n2 > t:
        res += len(arr)
    res += max(n1, n2)
    return res

arr2 = copy.deepcopy(arr)
anneal(arr2, t, fitness)

arr2 = copy.deepcopy(arr)
genetic(arr2, t, fitness)

arr2 = copy.deepcopy(arr)
climb(arr2, t, fitness)