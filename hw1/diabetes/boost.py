import csv
from process_data import process
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

file = open('../data/pima-indians-diabetes-database/diabetes.csv')

all_data = list(csv.reader(file))
data_size = len(all_data) - 1

val_frac = 0.1
test_frac = 0.2

max_depths = [1, 2, 3, 5, 10]
n_estimators = [1, 2, 3, 5, 10]

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
    best_acc = 0
    optimal_depth = None
    optimal_est = None
    optimal_clf = None
    for depth in max_depths:
        for est in n_estimators:
            clf = AdaBoostClassifier(base_estimator = DT(max_depth = depth), n_estimators = est)
            ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = process(all_data, train_frac, val_frac, test_frac)
            
            clf.fit(x_train, y_train)
            accuracy = accuracy_score(clf.predict(x_val), y_val)
            if accuracy > best_acc:
                best_acc = accuracy
                optimal_depth = depth
                optimal_est = est
                optimal_clf = clf
    train_performance.append(accuracy_score(optimal_clf.predict(x_train), y_train))
    val_performance.append(accuracy_score(optimal_clf.predict(x_val), y_val))
    test_performance.append(accuracy_score(optimal_clf.predict(x_test), y_test))


f1, ax1 = plt.subplots()
train_sizes = list(map(lambda frac: int(frac * data_size), train_fracs))
print(len(train_sizes), len(train_performance), len(val_performance), len(test_performance))
plt.xscale('log')
plt.plot(train_sizes, train_performance)
plt.plot(train_sizes, val_performance)
plt.plot(train_sizes, test_performance)
plt.legend(["Train", "Validation", "Test"])
plt.title("Diabetes: Boosted decision tree performance")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.savefig('diabetes_boost_trainingsize.png')