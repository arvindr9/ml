import copy
from anneal import FourPeaksAnneal
from hill_climb import climb
from genetic import genetic

t = 4
l = 10

arr = [0 for i in range(l)]
for i in range(2, l, 2):
    arr[i] = 1

def fitness(arr):
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

def convert(arr):
    arr2 = []
    for i in arr:
        arr2.append(int(round(i)))
    return arr2

def fitness_gen(arr):
    return fitness(convert(arr))




print("Simulated annealing:")
arr2 = copy.deepcopy(arr)
anneal = FourPeaksAnneal(arr2, fitness)
opt_arr, opt_fit = anneal.anneal()
print(opt_arr)
print(opt_fit)

arr = [0 for i in range(l)]
for i in range(2, l, 2):
    arr[i] = 1

print("Hill climbing")
arr2 = copy.deepcopy(arr)
opt_arr, opt_fit = climb(arr2, fitness)
print(opt_arr)
print(opt_fit)

arr = [0 for i in range(l)]
for i in range(2, l, 2):
    arr[i] = 1

print("Genetic algo:")
arr2 = copy.deepcopy(arr)
opt_arr, opt_fit = genetic(arr2, fitness_gen)
print(convert(opt_arr))
print(opt_fit)

