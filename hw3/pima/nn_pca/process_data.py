import numpy as np
import pickle
import csv


def getXy(data):
    X = []
    y = []
    for line in data:
        X.append(list(map(float,line[:-1])))
        y.append(list(map(float,(line[-1]))))
    return (np.array(X), np.array(y))


def process_data(data, train_frac, val_frac, test_frac):
    data = np.array(data)
    indices = np.random.permutation(np.arange(len(data)))
    data = data[indices]

    train_len = int(train_frac * len(data))
    val_len = int(val_frac * len(data))
    test_len = int(test_frac * len(data))

    train_data = data[:train_len]
    val_data = data[train_len: train_len + val_len]
    test_data = data[train_len + val_len: train_len + val_len + test_len]
    return getXy(train_data), getXy(val_data), getXy(test_data)

def process(data, train_frac, val_frac, test_frac, modify = False):
    data = data[1:]
    return process_data(data, train_frac, val_frac, test_frac)


    