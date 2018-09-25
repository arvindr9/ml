import csv
from process_data import process
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier as NN
import matplotlib.pyplot as plt
import numpy as np

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1

val_frac = 0.1
test_frac = 0.2

hid_layers = [(10,), (5, 10, 5,), (5, 10, 10, 5), (5, 10, 10, 10, 5,)]

train_fracs = [0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
#[0.0002, 0.0003, 0.0004, 0.0005, 

# for evaluating against train size
train_performance = []
val_performance = []
test_performance = []

features = None
encodings = None

for train_frac in train_fracs:
    acc = []
    clfs = []
    for hid_layer_specific in hid_layers:
        clf = NN(hid_layer_specific, activation = 'relu')
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = process(all_data, train_frac, val_frac, test_frac)
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_val = x_val.astype('float32')
        y_val = y_val.astype('float32')
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')
        
        clf.fit(x_train, y_train)
        clfs.append(clf)
        acc.append(accuracy_score(clf.predict(x_val), y_val))
    optimal_index = acc.index(max(acc))
    optimal_hid_layers = hid_layers[optimal_index]
    optimal_clf = clfs[optimal_index]
    train_performance.append(accuracy_score(optimal_clf.predict(x_train), y_train))
    val_performance.append(accuracy_score(optimal_clf.predict(x_val), y_val))
    test_performance.append(accuracy_score(optimal_clf.predict(x_test), y_test))

print(train_performance)
print(val_performance)
print(test_performance)

f1, ax1 = plt.subplots()
train_sizes = list(map(lambda frac: int(frac * data_size), train_fracs))
print(len(train_sizes), len(train_performance), len(val_performance), len(test_performance))
plt.xscale('log')
plt.plot(train_sizes, train_performance)
plt.plot(train_sizes, val_performance)
plt.plot(train_sizes, test_performance)
plt.legend(["Train", "Validation", "Test"])
plt.title("Diabetes: Neural network performance")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.savefig('diabetes_nn_trainingsize.png')

train_frac = 0.6

max_iterations = [2, 5, 10, 50, 100, 200]

# for evaluating over time
train_performance = []
val_performance = []
test_performance = []

for max_iter in max_iterations:
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = process(all_data, train_frac, val_frac, test_frac)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_val = x_val.astype('float32')
    y_val = y_val.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    clf = NN(max_iter = max_iter)
    clf.fit(x_train, y_train)
    train_performance.append(accuracy_score(clf.predict(x_train), y_train))
    val_performance.append(accuracy_score(clf.predict(x_val), y_val))
    test_performance.append(accuracy_score(clf.predict(x_test), y_test))

f2, ax2 = plt.subplots()
plt.xscale('log')
plt.plot(max_iterations, train_performance)
plt.plot(max_iterations, val_performance)
plt.plot(max_iterations, test_performance)
plt.legend(["Train", "Validation", "Test"])
plt.title("Diabetes: Neural Network performance vs iterations")
plt.xlabel("Max iterations")
plt.ylabel("Accuracy")
plt.savefig('diabetes_nn_iterations.png')