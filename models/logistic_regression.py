import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def log_likelihood(y, y_hat):
    return np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))


class LogisticRegression:
    """Classification model based on the logistic function.

    Classification model that maps a linear combination of the
    input features to a probability using the logistic function.

    Parameters
    ----------
    n_features : `int`
        Dimensionality, or number of features, of the training data.

    """

    def __init__(self, n_features):
        self.n_features = n_features
        self.w = self._initialize_weights(self.n_features+1)

    def _initialize_weights(self, n_features):
        return np.zeros(n_features)

    def fit(self, X, y, learning_rate=1e-3, iterations=1000):
        """Trains the model with gradient descent.

        Parameters
        ----------
        x : `numpy.ndarray` (n_examples, n_features)
            Independant variables of the training examples from which to
            estimate the parameters of the model.

        y : `numpy.ndarray` (n_examples, n_features)
            Labels corresponding to each training example in ``x``.

        learning_rate : `int`, optional
            Size of the step to take using the negative of the gradient.
            Defaults to 1e-3

        iterations : `int`, optional
            Number of iterations to perform gradient descent. Defaults to 1000.

        Returns
        -------
        loss : `list`
            List containing the loss on the training set at each iteration

        acc : `list`
            List containing the accuracy on the training set at each iteraton

        """

        # Add a first dimension of ones corresponding to the intercept
        x = np.hstack((np.ones([X.shape[0], 1]), X))

        loss = []
        acc = []
        for _ in range(iterations):
            # Forward pass
            z = np.matmul(x, self.w)
            y_hat = sigmoid(z)

            loss.append(self.loss(y, y_hat))
            acc.append(np.mean(np.around(y_hat) == y))

            # Gradients
            dz = y_hat - y
            dw = np.matmul(x.T, dz)

            self.w -= learning_rate*dw

        return loss, acc

    def predict(self, X):
        """Predicts the response using the current estimands.

        Parameters
        ----------
        X : `numpy.ndarray` (n_examples, n_features)
            Data from where to infer a prediction

        Returns
        -------
        predicted : `numpy.ndarray` (n_examples,)
            Predicted probabilities for each data example

        """

        X = np.atleast_2d(X)
        X = np.reshape(X, [-1, self.n_features])

        # Add a first dimension of ones corresponding to the intercept
        X = np.hstack((np.ones([X.shape[0], 1]), X))
        return sigmoid(np.matmul(X, self.w))

    def loss(self, y, y_hat):
        # Return loss to minimize by gradient descent.
        return -(1.0/len(y))*log_likelihood(y, y_hat)
