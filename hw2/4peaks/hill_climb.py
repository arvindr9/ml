import random
import copy

def climb(arr, fitness, out_iter = 10, in_iter = 100, n_sample = 10):
    
    max_out_score = fitness(arr)
    global_optimum = arr
    for i in range(out_iter):
        arr2 = copy.deepcopy(arr)
        for j in range(in_iter):
            maxScore = fitness(arr2)
            best_sample = arr2
            for sample in range(n_sample):
                arr3 = copy.deepcopy(arr2)
                randomi = random.randint(0, len(arr3) - 1)
                randomc = random.randint(0, 1)
                arr3[randomi] = randomc
                score = fitness(arr3)
                if score > maxScore:
                    maxScore = score
                    best_sample = arr3
            arr2 = best_sample
        score = fitness(arr2)
        if score > max_out_score:
            max_out_score = score
            global_optimum = arr2
    return global_optimum, max_out_score

