import numpy as np
from collections import Counter


def similarity(x, X):
    xtiled = np.tile(np.reshape(x, [1, -1]), (X.shape[0], 1))
    distances = np.sum(np.square(xtiled-X), axis=1)
    return distances


class KNN:
    """Non-parametric model that can be used either for classification or
    regression.

    """

    def __init__(self, X, y, k=5):
        self.X = X
        self.y = y
        self.k = k

    def _predict_single(self, x):
        d = similarity(x, self.X)
        nearest_labels = self.y[np.argsort(d)[:self.k]]
        y_hat, _ = Counter(nearest_labels).most_common(1)[0]

        return y_hat

    def predict(self, x):
        x = np.atleast_2d(x)
        x = np.reshape(x, [-1, self.X.shape[1]])
        y_hat = np.zeros(x.shape[0])
        for i in range(len(y_hat)):
            y_hat[i] = self._predict_single(x[i, :])

        return y_hat
