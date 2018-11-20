#initial clustering

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans



import matplotlib.pyplot as plt
import numpy as np
from process_data import process

from sklearn.metrics import accuracy_score

import csv

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1



train_frac = 1

(x_train, y_train), _, _ = process(all_data, train_frac, 0, 0)

n_samples = 200

print("GMM")
# print(y_train[0])
gmm = GaussianMixture(n_components=3)
gmm.fit(x_train)
print(gmm.predict(x_train[:n_samples]))
# print(s)
# print(x_train[0])
# print(y_train[0])

print("K-means")
km = KMeans(n_clusters = 3)
km.fit(x_train)
print(km.predict(x_train[:n_samples]))

print("y_train")
print(y_train[:n_samples].reshape(1, -1))

print("Elbow method for GMM")

aic = []
bic = []
for i in range(1, 11):
        gmm = GaussianMixture(n_components = i)
        gmm.fit(x_train)
        aic.append(gmm.aic(x_train))
        bic.append(gmm.bic(x_train))

fig, ax = plt.subplots()
plt.title("Initial clustering: GMM")
plt.xlabel("number of components")
plt.plot(range(1, 11), aic)
plt.plot(range(1, 11), bic)
plt.legend(["AIC", "BIC"])
plt.savefig("gmm_init.png")


print("Elbow method for k-Means")

wcss = []

for i in range(1, 11):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(x_train)
        wcss.append(kmeans.inertia_)

fig2, ax2 = plt.subplots()
plt.title("Initial clustering: K-Means")
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.plot(range(1, 11), wcss)
plt.savefig("km_init.png")

