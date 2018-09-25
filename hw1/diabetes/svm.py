from sklearn.svm import SVC
from process_data import process
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# from id3 import Id3Estimator, export_graphviz
import csv

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

# clf = Id3Estimator(max_depth = 10)

all_data = list(csv.reader(file))
data_size = len(all_data) - 1
# train_frac = 0.6
val_frac = 0.1
test_frac = 0.2


kernel_funcs = ['linear', 'rbf']

train_fracs = [0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]

# for evaluating against train size
train_performance = []
val_performance = []
test_performance = []



features = None
encodings = None
for train_frac in train_fracs:
    acc = []
    clfs = []
    for kernel in kernel_funcs:
        clf = SVC(kernel = kernel)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = process(all_data, train_frac, val_frac, test_frac)
        print(x_train.shape, y_train.shape)
        clf.fit(x_train, y_train)
        clfs.append(clf)
        acc.append(accuracy_score(clf.predict(x_val), y_val))
    optimal_index = acc.index(max(acc))
    optimal_kernel = kernel_funcs[optimal_index]
    optimal_clf = clfs[optimal_index]
    train_performance.append(accuracy_score(optimal_clf.predict(x_train), y_train))
    val_performance.append(accuracy_score(optimal_clf.predict(x_val), y_val))
    test_performance.append(accuracy_score(optimal_clf.predict(x_test), y_test))

print(train_performance)
print(val_performance)
print(test_performance)



'''
hyperparameters: max depth, number of estimators
plot: performance vs training size
plot: performance vs time
'''
f1, ax1 = plt.subplots()
train_sizes = list(map(lambda frac: int(frac * data_size), train_fracs))
print(len(train_sizes), len(train_performance), len(val_performance), len(test_performance))
plt.xscale('log')
plt.plot(train_sizes, train_performance)
plt.plot(train_sizes, val_performance)
plt.plot(train_sizes, test_performance)
plt.legend(["Train", "Validation", "Test"])
plt.title("Diabetes: SVM performance")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.savefig('diabetes_svm_trainingsize.png')

train_frac = 0.6

max_iterations = [10, 50, 100, 200, 300, 500, 1000, 2000]

#For evaluating against time
train_performance = []
val_performance = []
test_performance = []


for max_iter in max_iterations:
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = process(all_data, train_frac, val_frac, test_frac)
    clf = SVC(max_iter = max_iter)
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
plt.title("Diabetes: SVM performance vs iterations")
plt.xlabel("Max iterations")
plt.ylabel("Accuracy")
plt.savefig('diabetes_svm_iterations.png')