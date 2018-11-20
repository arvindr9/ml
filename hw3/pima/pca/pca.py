from sklearn.decomposition import PCA
from process_data import process
import csv

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1



train_frac = 1
val_frac = 0.2
test_frac = 0.2

(x_train, y_train), _, _= process(all_data, train_frac, 0, 0)

pca = PCA(n_components = 2)
pca.fit(x_train)
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
reduced = np.array(pca.transform(x_train))
x = reduced[:, 0]
y = reduced[:, 1]
colors = []
eps = 1e-4
for elt in y_train:
    if abs(1 - elt) < eps:
        colors.append('r')
    else:
        colors.append('b')

fig, ax = plt.subplots()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.scatter(x, y, c = colors)
plt.savefig("diabetes_pca.png")

print(reduced)
print(y_train)

with open("reduced_data.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(reduced)
with open("labels.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(y_train)