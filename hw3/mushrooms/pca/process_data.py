import numpy as np
import pickle
import csv

encoding_list = [{'p': 0, 'e': 1}]



def create_encodings(data, features):
    for col in range(1, len(features) + 1):
        encoding_dict = {}
        code = 0
        for row in range(len(data)):
            if data[row][col] in encoding_dict:
                continue
            encoding_dict[data[row][col]] = code
            code += 1
        encoding_list.append(encoding_dict)

def getXy(data):
    X = []
    y = []
    for line in data:
        X.append(line[1:])
        y.append(line[0])
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

def convertData(data):
    cnt = 0
    for i in range(len(data)):
        cnt += 1
        for j in range(len(data[0])):
            if i == 0:
                print(i, j, data[i][j])
            data[i][j] = encoding_list[j][data[i][j]]




def process(data, train_frac, val_frac, test_frac, modify = False, features = None, encodings = None):
    if modify:
        features = data[0][1:]
        data = data[1:]
        create_encodings(data, features)
        convertData(data)
    else:
        data = data[1:]
    return features, encodings, process_data(data, train_frac, val_frac, test_frac)


    