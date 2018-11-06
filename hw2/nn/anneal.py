# def f(z, *params):
#     x, y = z
#     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
#     return f1(z, *params) + f2(z, *params) + f3(z, *params)
# import numpy as np
# x0 = np.array([2., 2.])     # Initial guess.
# from scipy import optimize
# np.random.seed(555)   # Seeded to allow replication.
# res = optimize.anneal(f, x0, args=params, schedule='boltzmann', full_output=True, maxiter=500, lower=-10,
#                           upper=10, dwell=250, disp=True)
# import numpy as np

# def anneal(clf, hid_layers, x_train, y_train):
#     clf.coefs_.append(np.zeros((x_train.shape[1], hid_layers[0])))
#     for i in range(len(hid_layers) - 1):
#         clf.coefs_.append(np.zeros((hid_layers[i], hid_layers[i - 1])))
#         clf.intercepts_.append(np.zeros((hid_layers[i],)))
#     clf.coefs_.append(np.zeros((hid_layers[len(hid_layers) - 1], 1)))
#     clf.intercepts_.append(np.zeros((hid_layers[len(hid_layers) - 1],)))
#     clf.intercepts_.append(np.zeros((1,)))

from simanneal import Annealer
import numpy as np
from sklearn.metrics import accuracy_score

class NNAnneal(Annealer):
    def __init__(self, clf, hid_layers, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = clf
        clf.coefs_.append(np.zeros((x_train.shape[1], hid_layers[0])))
        for i in range(len(hid_layers) - 1):
            clf.coefs_.append(np.zeros((hid_layers[i], hid_layers[i - 1])))
            clf.intercepts_.append(np.zeros((hid_layers[i],)))
        clf.coefs_.append(np.zeros((hid_layers[len(hid_layers) - 1], 1)))
        clf.intercepts_.append(np.zeros((hid_layers[len(hid_layers) - 1],)))
        clf.intercepts_.append(np.zeros((1,)))
        super(NNAnneal, self).__init__([clf.coefs_, clf.intercepts_])
    def move(self):
        for i in range(len(self.state[0])):
            self.state[0][i] += 0.5 * np.random.randn(self.state[0][i].shape[0], self.state[0][i].shape[1])
        for i in range(len(self.clf.intercepts_)):
            self.state[1][i] += 0.5 * np.random.randn(self.state[1][i].shape[0],)
            # print(self.state[1][i])
            # print(self.clf.intercepts_[i])
    def energy(self):
        tmp_coef = self.clf.coefs_
        tmp_inter = self.clf.intercepts_
        self.clf.coefs_ = self.state[0]
        self.clf.intercepts_ = self.state[1]
        res = -accuracy_score(self.clf.predict(self.x_train), self.y_train)
        self.clf.coefs_ = tmp_coef
        self.clf.intercepts_ = tmp_inter
        return res 