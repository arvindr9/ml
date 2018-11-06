libraries for ML algos:
Simulated annealing: scipy.optimize.anneal
Genetic algo: scipy.optimize.differential_evolution
Randomized hill climbing: ?

neural network:
W1, b1, W2, b2 (2 layers can be done for now)
To test different weights, use the coefs_ and intercepts_ attributes: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

four peaks: https://www.cc.gatech.edu/~isbell/tutorials/mimic-tutorial.pdf

k-coloring: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15451-s00/www/lectures/lect0323post.txt

TODO:
create nn on the mnist
create the algorithms s.t. the nn can be trained on the mnist
find new datasets
Write an analysis of the algos on the two datasets that have the most differing results

Data:
contains the Kaggle Mushrooms dataset

Folders:
nn: Neural network scripts
4peaks: Four peaks scripts
kcolor: K-color scripts

Scripts in each folder:
main script (e.g. nn.py, 4peaks.py, etc..) Declares the problem and calls other algorithms
anneal.py: Simulated annealing algorithm
hill_climb.py: Randomized hill-climbing algorithm
genetic.py: Genetic algorithm