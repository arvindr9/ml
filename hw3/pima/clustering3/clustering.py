#clustering to reduce dimensions

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
from process_data import process

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier as MLP

import csv

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1

train_frac = 0.6
val_frac = 0.2
test_frac = 0.2

(x_train, y_train), (x_val, y_val), (x_test, y_test) = process(all_data, train_frac, val_frac, test_frac)

n_samples = 200
maxc = 20 #maximum number of clusters plus 1

y_train = y_train.flatten()
print(y_train.shape)

nn = MLP()
nn.fit(x_train, y_train)
real_train_acc = accuracy_score(nn.predict(x_train), y_train)
real_test_acc = accuracy_score(nn.predict(x_test), y_test)

print("GMM")
# print(y_train[0])
# gmm = GaussianMixture(n_components=3)
# gmm.fit(x_train)
# gmm_x_train = gmm.predict_proba(x_train)
# gmm_x_val = gmm.predict_proba(x_val)
# gmm_x_test = gmm.predict_proba(x_test)

# print(gmm_x_train)
# print(y_train)

train_acc = []
test_acc = []
for i in range(1, maxc):
    gmm = GaussianMixture(n_components = i)
    gmm.fit(x_train)
    gmm_x_train = gmm.predict_proba(x_train)
    gmm_x_val = gmm.predict_proba(x_val)
    gmm_x_test = gmm.predict_proba(x_test)
    gmm_nn = MLP()
    gmm_nn.fit(gmm_x_train, y_train)
    train_acc.append(accuracy_score(gmm_nn.predict(gmm_x_train), y_train))
    test_acc.append(accuracy_score(gmm_nn.predict(gmm_x_test), y_test))

fig, ax = plt.subplots()

plt.title("GMM")
plt.xlabel("Number of clusters")
plt.ylabel("Accuracy")
plt.plot(range(1, maxc), train_acc)
plt.plot(range(1, maxc), test_acc)
plt.plot(range(1, maxc), [real_train_acc] * (maxc - 1))
plt.plot(range(1, maxc), [real_test_acc] * (maxc - 1))
plt.legend(["Training accuracy", "Testing accuracy", "Original training accuracy", "Original testing accuracy"])
plt.savefig("gmm_acc.png")



# gmm_nn = MLP()
# gmm_nn.fit(gmm_x_train, y_train)
# print("Train accuracy:", accuracy_score(gmm_nn.predict(gmm_x_train), y_train))
# print("Test accuracy:", accuracy_score(gmm_nn.predict(gmm_x_test), y_test))


print("K-means")

train_acc = []
test_acc = []

for i in range(1, maxc):
    km = KMeans(n_clusters = i)
    km.fit(x_train)
    km_x_train = km.transform(x_train)
    km_x_val = km.transform(x_val)
    km_x_test = km.transform(x_test)
    km_nn = MLP()
    km_nn.fit(km_x_train, y_train)
    train_acc.append(accuracy_score(km_nn.predict(km_x_train), y_train))
    test_acc.append(accuracy_score(km_nn.predict(km_x_test), y_test))

# print("Train accuracy:", accuracy_score(km_nn.predict(km_x_train), y_train))
# print("Test accuracy:", accuracy_score(km_nn.predict(km_x_test), y_test))

fig2, ax2 = plt.subplots()

plt.title("K-means")
plt.xlabel("Number of clusters")
plt.ylabel("Accuracy")
plt.plot(range(1, maxc), train_acc)
plt.plot(range(1, maxc), test_acc)
plt.plot(range(1, maxc), [real_train_acc] * (maxc - 1))
plt.plot(range(1, maxc), [real_test_acc] * (maxc - 1))
plt.legend(["Training accuracy", "Testing accuracy", "Original training accuracy", "Original testing accuracy"])
plt.savefig("km_acc.png")
