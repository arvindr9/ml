#clustering on data with reduced dimensions

import csv
from numpy import genfromtxt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x_train = genfromtxt("../pca/reduced_data.csv", delimiter = ',')
y_train = genfromtxt("../pca/labels.csv")
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

for i in [2, 4, 6]:

    print("KMeans cluster display:" + str(i))
    y_pred = KMeans(n_clusters=i).fit_predict(x_train)
    f, ax = plt.subplots()
    plt.title("K means")
    plt.scatter(x_train[:, 0], x_train[:, 1], c = y_pred)
    plt.savefig("cluster_{}_km_pca.png".format(i))

    print("GMM cluster display:" + str(i))
    y_pred = GaussianMixture(n_components=i).fit_predict(x_train)
    f, ax = plt.subplots()
    plt.title("GMM")
    plt.scatter(x_train[:, 0], x_train[:, 1], c = y_pred)
    plt.savefig("cluster_{}_gmm_pca.png".format(i))