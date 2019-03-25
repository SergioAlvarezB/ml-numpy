import numpy as np
from collections import Counter


def euclidean_distance(x, X):
    xtiled = np.tile(np.reshape(x, [1, -1]), (X.shape[0], 1))
    distances = np.sum(np.square(xtiled-X), axis=1)
    return distances


class KNNBase:
    """Base class for the KNN model.

    Parameters
    ----------
    X : `numpy.ndarray` (n_samples, n_features)
        Set of known points that will be used for inference.

    y : `numpy.ndarray` (n_camples,)
        Vector of labels corresponding to each point in `X`. Labels can be
        integers indicating the class of each point in the case of the
        classification task, or real values for regression.

    k : `int`, optional
        Number of neighbors to use to make a prediction. Defaults to 5.

    distance : `str`, optional
        Metric to use to compute the distance between points. Defaults to
        "euclidean".

    """

    def __init__(self, X, y, k=5, distance="euclidean"):
        self.X = X
        self.y = y
        self.k = k
        self.distance = self._get_distance(distance)

    def _get_distance(self, type):
        # Return the distance metric as a callable
        if type == "euclidean":
            return euclidean_distance

    def _get_neighbors(self, x):
        d = self.distance(x, self.X)
        nearest_idx = np.argsort(d)[:self.k]
        return nearest_idx


class KNNClassifier(KNNBase):
    """Classifier version of the KNN.

    KNNClassifier is a non-parametric model that estimates the label of a new
    point as the majority label of its k-closests neighbors in the train set.
    """

    def _predict_single(self, x):
        nearest_idx = self._get_neighbors(x)
        nearest_labels = self.y[nearest_idx]
        y_hat, _ = Counter(nearest_labels).most_common(1)[0]

        return y_hat

    def predict(self, x):
        x = np.atleast_2d(x)
        x = np.reshape(x, [-1, self.X.shape[1]])
        y_hat = np.zeros(x.shape[0])
        for i in range(len(y_hat)):
            y_hat[i] = self._predict_single(x[i, :])

        return y_hat


class KNNRegressor(KNNBase):
    """Regression version of the KNN.

    KNNRegression is a non-parametric model that estimates the value of a new
    point as the average its k-closests neighbors in the train set.
    """

    def _predict_single(self, x):
        nearest_idx = self._get_neighbors(x)
        y_hat = np.mean(self.y[nearest_idx])

        return y_hat

    def predict(self, x):
        x = np.atleast_2d(x)
        x = np.reshape(x, [-1, self.X.shape[1]])
        y_hat = np.zeros(x.shape[0])
        for i in range(len(y_hat)):
            y_hat[i] = self._predict_single(x[i, :])

        return y_hat
