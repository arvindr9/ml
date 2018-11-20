from sklearn.decomposition import PCA
from process_data import process
import csv
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1

max_dim = 8 #max number of dimensions plus 1



train_frac = 0.6
val_frac = 0.2
test_frac = 0.2

(x_train, y_train), (x_val, y_val), (x_test, y_test) = process(all_data, train_frac, val_frac, test_frac)

nn = MLP()
print(x_train.shape)
print(y_train.shape)
nn.fit(x_train, y_train)
real_train_acc = accuracy_score(nn.predict(x_train), y_train)
real_test_acc = accuracy_score(nn.predict(x_test), y_test)
train_acc = []
test_acc = []

for dim in range(1, max_dim):
    pca = PCA(n_components = dim)
    pca.fit(x_train)
    reduced_x_train = np.array(pca.transform(x_train))
    reduced_x_test = np.array(pca.transform(x_test))
    nn = MLP()
    nn.fit(reduced_x_train, y_train)
    train_acc.append(accuracy_score(nn.predict(reduced_x_train), y_train))
    test_acc.append(accuracy_score(nn.predict(reduced_x_test), y_test))



fig, ax = plt.subplots()
plt.title("PCA: Neural network performance")
plt.xlabel("Number of dimensions")
plt.ylabel("Accuracy")
plt.plot(range(1, max_dim), [real_train_acc] * (max_dim - 1))
plt.plot(range(1, max_dim), [real_test_acc] * (max_dim - 1))
plt.plot(range(1, max_dim), train_acc)
plt.plot(range(1, max_dim), test_acc)
plt.legend(["Original training accuracy", "Original testing accuracy", "Training accuracy", "Testing accuracy"])
plt.savefig("nn_pca.png")

# print(reduced)
# print(y_train)

# with open("reduced_data.csv", "w") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(reduced)
# with open("labels.csv", "w") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(y_train)