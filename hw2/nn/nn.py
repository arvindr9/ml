# import tensorflow as tf
# mnist = tf.keras.datasets.mnist
# from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPClassifier as NN
# import matplotlib.pyplot as plt
# import numpy as np

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# hid_layers = (784, 200, 100)

# clf = NN(hid_layers, activation = 'relu')
# print(clf.predict(x_train[0]))

from anneal import NNAnneal
from genetic import genetic
from hill_climb import climb
import csv
from process_data import process
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier as NN
import matplotlib.pyplot as plt
import numpy as np

file = open('../data/mushroom-classification/mushrooms.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1

val_frac = 0.1
test_frac = 0.2

hid_layers = (5,)

train_frac = 0.7
#[0.0002, 0.0003, 0.0004, 0.0005, 

# for evaluating against train size
train_performance = []
val_performance = []
test_performance = []

features = None
encodings = None


clf = NN(hid_layers, activation = 'relu')
features, encodings, ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = process(all_data, train_frac, val_frac, test_frac, modify = True)
print(x_train.shape, y_train.shape)
clf.fit(x_train, y_train)


print("Simulated annealing:")
acc_anneal = []
test_anneal = []

clf.coefs_ = []
clf.intercepts_ = []
#anneal(clf, hid_layers, x_train, y_train) #uses simulated annealing to find the optimal weights
anneal = NNAnneal(clf, hid_layers, x_train, y_train)
([clf.coefs_, clf.intercepts_]), e = anneal.anneal()

print(accuracy_score(clf.predict(x_train), y_train))
print(accuracy_score(clf.predict(x_val), y_val))
print(accuracy_score(clf.predict(x_test), y_test))
# acc_anneal.append(accuracy_score(clf.predict(x_val), y_val))
# test_anneal.append(accuracy_score(clf.predict(x_test), y_test))

# print(acc_anneal)
# print(test_anneal)

print("Hill climbing:")
clf.coefs_ = []
clf.intercepts_ = []

climb(clf, hid_layers, x_train, y_train)
print(accuracy_score(clf.predict(x_train), y_train))
print(accuracy_score(clf.predict(x_val), y_val))
print(accuracy_score(clf.predict(x_test), y_test))

print("Genetic algorithm:")
clf.coefs_ = []
clf.intercepts_ = []
genetic(clf, hid_layers, x_train, y_train)



