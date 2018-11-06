import numpy as np
from scipy.optimize import differential_evolution

clf = []
x_train = []
y_train = []

def findScore(coefs, intercepts, clf, x_train, y_train):
    tmp_coef = clf.coefs_
    tmp_inter = clf.intercepts_
    clf.coefs_ = coefs
    clf.intercepts_ = intercepts
    res = accuracy_score(clf.predict(x_train), y_train)
    clf.coefs_ = tmp_coef
    clf.intercepts_ = tmp_inter
    return -res

def genetic(clf_orig, hid_layers, x_train_orig, y_train_orig):
    clf = clf_orig
    x_train = x_train_orig
    y_train = y_train_orig
    clf.coefs_.append(np.zeros((x_train.shape[1], hid_layers[0])))
    for i in range(len(hid_layers) - 1):
        clf.coefs_.append(np.zeros((hid_layers[i], hid_layers[i - 1])))
        clf.intercepts_.append(np.zeros((hid_layers[i],)))
    clf.coefs_.append(np.zeros((hid_layers[len(hid_layers) - 1], 1)))
    clf.intercepts_.append(np.zeros((hid_layers[len(hid_layers) - 1],)))
    clf.intercepts_.append(np.zeros((1,)))
    bounds = [(-5, 5), (-5, 5)]
    result = differential_evolution(findScore, bounds)
    print("Coeffs and intercepts")
    print(result.x)
    print("accuracy score")
    print(-result.fun)
