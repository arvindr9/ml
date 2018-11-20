import csv
from process_data import process
from sklearn.decomposition import PCA


file = open('../data/mushroom-classification/mushrooms.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1

train_frac = 1.0

features = None
encodings = None

features, encodings, ((x_train, y_train), _, _) = process(all_data, train_frac, 0, 0, modify = True)

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
plt.savefig("mushrooms_pca.png")

print(reduced)

with open("reduced_data.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(reduced)
with open("labels.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(y_train)

pca = PCA(n_components = 3)
pca.fit(x_train)
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
reduced = np.array(pca.transform(x_train))
x = reduced[:, 0]
y = reduced[:, 1]
z = reduced[:, 2]
df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
colors = []
eps = 1e-4
for elt in y_train:
    if abs(1 - elt) < eps:
        colors.append('r')
    else:
        colors.append('b')

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
# plt.xlabel("Feature 1")
# ax.ylabel("Feature 2")
# ax.zlabel("Feature 3")
plt.scatter(x, y, z, c = colors)
ax.view_init(azim = 20)
plt.savefig("mushrooms_pca_3d.png")

print(reduced)

with open("reduced_data_3d.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(reduced)
with open("labels_3d.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(y_train)