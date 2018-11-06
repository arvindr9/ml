import random
import copy

def climb(colors, k, edges, fitness, out_iter = 10, in_iter = 100, n_sample = 10):
    
    min_out_score = fitness(colors)
    global_optimum = colors
    for i in range(out_iter):
        colors2 = copy.deepcopy(colors)
        for j in range(in_iter):
            minScore = fitness(colors2)
            best_sample = colors2
            for sample in range(n_sample):
                colors3 = copy.deepcopy(colors2)
                randomi = random.randint(0, len(colors3) - 1)
                randomc = random.randint(1, k)
                colors3[randomi] = randomc
                score = fitness(colors3)
                if score < minScore:
                    minScore = score
                    best_sample = colors3
            colors2 = best_sample
        score = fitness(colors2)
        if score < min_out_score:
            min_out_score = score
            global_optimum = colors2
    return global_optimum, min_out_score