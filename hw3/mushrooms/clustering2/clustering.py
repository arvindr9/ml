#Clustering after running pca

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans



import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
from process_data import process

from sklearn.metrics import accuracy_score

import csv

x_train = genfromtxt("../pca/reduced_data.csv", delimiter = ',')
y_train = genfromtxt("../pca/labels.csv", delimiter = ',')
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
plt.title("Clustering after PCA: GMM")
plt.xlabel("number of components")
plt.plot(range(1, 11), aic)
plt.plot(range(1, 11), bic)
plt.legend(["AIC", "BIC"])
plt.savefig("gmm_pca.png")


print("Elbow method for k-Means")

wcss = []

for i in range(1, 11):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(x_train)
        wcss.append(kmeans.inertia_)

fig2, ax2 = plt.subplots()
plt.title("Clustering after PCA: K-Means")
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.plot(range(1, 11), wcss)
plt.savefig("km_pca.png")

print("KMeans cluster display")
y_pred = KMeans(n_clusters=6).fit_predict(x_train)
f, ax = plt.subplots()
plt.title("K means")
plt.scatter(x_train[:, 0], x_train[:, 1], c = y_pred)
plt.savefig("cluster_km_pca.png")

print("GMM cluster display")
y_pred = GaussianMixture(n_components=6).fit_predict(x_train)
f, ax = plt.subplots()
plt.title("GMM")
plt.scatter(x_train[:, 0], x_train[:, 1], c = y_pred)
plt.savefig("cluster_gmm_pca.png")