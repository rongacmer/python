import numpy as np

class rbf:
    def __int__(self):
        pass

    def predict(self, X):
        gamma = X.shape[1]
        X = np.mat(X)
        X = np.sum(X, axis = 1)
        X = X / gamma
        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y = np.exp(-X)
        y = y.A
        return y

    def predictx(self,X):
        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(10)
        return y

    def fit(self, x, y, sample_weight = None):
        pass
# a = np.mat([[1,2],[3,4]])
# clf = rbf()
# y = clf.predict(a)
