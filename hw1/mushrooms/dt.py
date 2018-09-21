from id3 import Id3Estimator
import csv

# library: https://pypi.org/project/decision-tree-id3/
# documentation: https://svaante.github.io/decision-tree-id3/

file = open('../data/mushroom-classification/mushrooms.csv')

class_encodings = {'p': 0, 'e': 1}
encoding_list = [{}]

clf = Id3Estimator(max_depth = 10, prune = True)

def getXy(data):
    X = []
    y = []
    for line in data:
        line_info = line.split()
        X.append(line_info[1:])
        y.append(class_encodings[line_info[0]])
    return (X, y)

def process_data(data, train_frac):
    create_encodings(data)
    train_len = int(train_frac * len(data))
    train_data = data[:train_len]
    test_data = data[train_len:]
    return getXy(train_data), getXy(test_data)

all_data = list(csv.reader(file))
features = all_data[0].split()[1:]
data = all_data[1:]
train_frac = 0.7


(x_train, y_train), (x_test, y_test) = process_data(data, train_frac)


