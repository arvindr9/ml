import csv
from process_data import process
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1

val_frac = 0.1
test_frac = 0.2

Ks = [1, 2, 3, 4, 5]

train_fracs = [0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
#[0.0002, 0.0003, 0.0004, 0.0005, 

# for evaluating against train size
train_performance = []
val_performance = []
test_performance = []


features = None
encodings = None

for train_frac in train_fracs:
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = process(all_data, train_frac, val_frac, test_frac)
    acc = []
    clfs = []
    for K in Ks:
        clf = KNN(n_neighbors = K)
        clf.fit(x_train, y_train)
        clfs.append(clf)
        acc.append(accuracy_score(clf.predict(x_val), y_val))
    optimal_index = acc.index(max(acc))
    optimal_K = Ks[optimal_index]
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
plt.title("Diabetes: KNN performance")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.savefig('diabetes_knn_trainingsize.png')