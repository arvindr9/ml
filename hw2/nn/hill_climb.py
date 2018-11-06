import numpy as np
import copy
from sklearn.metrics import accuracy_score

def findScore(coefs, intercepts, clf, x_train, y_train):
    tmp_coef = clf.coefs_
    tmp_inter = clf.intercepts_
    clf.coefs_ = coefs
    clf.intercepts_ = intercepts
    res = accuracy_score(clf.predict(x_train), y_train)
    clf.coefs_ = tmp_coef
    clf.intercepts_ = tmp_inter
    return res


def climb(clf, hid_layers, x_train, y_train, out_iter = 10, in_iter = 100, n_sample = 10):
    clf.coefs_.append(np.zeros((x_train.shape[1], hid_layers[0])))
    for i in range(len(hid_layers) - 1):
        clf.coefs_.append(np.zeros((hid_layers[i], hid_layers[i - 1])))
        clf.intercepts_.append(np.zeros((hid_layers[i],)))
    clf.coefs_.append(np.zeros((hid_layers[len(hid_layers) - 1], 1)))
    clf.intercepts_.append(np.zeros((hid_layers[len(hid_layers) - 1],)))
    clf.intercepts_.append(np.zeros((1,)))

    max_out_score = accuracy_score(clf.predict(x_train), y_train)
    global_optimum = [clf.coefs_, clf.intercepts_]

    for i in range(out_iter):
        coefs = copy.deepcopy(clf.coefs_)
        intercepts = copy.deepcopy(clf.intercepts_)

        for i_c in range(len(coefs)):
            coefs[i_c] = np.random.randn(coefs[i_c].shape[0], coefs[i_c].shape[1])
        for i_i in range(len(intercepts)):
            intercepts[i_i] = np.random.randn(intercepts[i_i].shape[0],)

        for j in range(in_iter):
            maxScore = findScore(coefs, intercepts, clf, x_train, y_train)
            bestSample = [coefs, intercepts]
            for sample in range(n_sample):
                coefs2 = []
                intercepts2 = []
                for i_c in range(len(coefs)):
                    coefs2.append(coefs[i_c] + 0.1 * np.random.randn(coefs[i_c].shape[0], coefs[i_c].shape[1]))
                for i_i in range(len(intercepts)):
                    intercepts2.append(intercepts[i_i] + 0.1 * np.random.randn(intercepts[i_i].shape[0],))
                score = findScore(coefs2, intercepts2, clf, x_train, y_train)
                if score > maxScore:
                    maxScore = score
                    bestSample = [coefs2, intercepts2]
            coefs, intercepts = bestSample
        score = findScore(coefs, intercepts, clf, x_train, y_train)
        if score > max_out_score:
            max_out_score = score
            global_optimum = [coefs, intercepts]
    clf.coefs_, clf.intercepts_ = global_optimum
    print(accuracy_score(clf.predict(x_train), y_train))
    

            

            

