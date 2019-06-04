import numpy as np

from utils import kernels


class SVM:
    """SVM classifier class.

    Implements a SVM classification model. The algorithm minimize the dual form
    cost using a projected version of gradient descent.

    Parameters
    ----------
    kernel : `str`, optional
        Kernel to use. Defaults to "linear".

    C : `float`, optional
        Cost parameter, specifies how much we penalize incorrectly classified
        points, the case C=inf is only possible with separable data.
        Defaults to 1.

    """
    def __init__(self, kernel='linear', C=1, degree=2, gamma=1.0/4):
        self.C = C
        self.kernel = self._get_kernel(kernel, degree, gamma)

    def _get_kernel(self, kernel, degree, gamma):
        # return kernel as a Callable
        if kernel == 'linear':
            return kernels.linear

        elif kernel == 'quadratic':
            return lambda x, y: kernels.polynomial(x, y, d=2)

        elif kernel == 'polynomial':
            return lambda x, y: kernels.polynomial(x, y, d=degree)

        elif kernel == 'rbf':
            return lambda x, y: kernels.rbf(x, y, gamma=gamma)

        else:
            raise NotImplemented

    def _fit_analytic(self, X, y):
        """Fits the lagr_multipliers with the analytic solution of the dual
        form. THIS METHOD IS NOT CORRECT, although KKT conditions can be
        enforced a posteriory, the solution is no longer guaranteed to be
        optimal.
        """
        # Transform labels from format {0, 1} to {-1, 1}
        y[y == 0] = -1

        n_samples, n_dimensions = X.shape

        # Compute kernel_matrix labeled
        G = np.zeros([n_samples, n_samples])

        for i in range(n_samples-1):
            for j in range(i, n_samples):
                G[i, j] = y[i]*y[j]*self.kernel(X[i, :], X[j, :])
                G[j, i] = G[i, j]

        # Compute analytic solution
        lagr_multipliers = np.sum(np.linalg.pinv(G), axis=0)
        lagr_multipliers = np.clip(lagr_multipliers, 0, self.C)

        supported = lagr_multipliers > 0
        self.lagr_multipliers = lagr_multipliers[supported]
        self.supported_vectors = X[supported, :]
        self.labels = y[supported]

        # Compute intercept
        bias = []
        for i in range(len(self.labels)):

            if self.lagr_multipliers[i] < self.C:
                b = self.labels[i]
                for j in range(len(self.labels)):
                    if i != j:
                        k = self.kernel(self.supported_vectors[j],
                                        self.supported_vectors[i])
                        b -= self.lagr_multipliers[j] * self.labels[j] * k
                bias.append(b)
        self.b = np.mean(np.array(bias))

    def fit(self, X, y, iterations=1000, learning_rate=0.01):

        # Transform labels from format {0, 1} to {-1, 1}
        y[y == 0] = -1

        n_samples, n_dimensions = X.shape

        # Initialize lagrange multipliers
        lagr_multipliers = np.random.rand(n_samples)/n_samples

        # Compute kernel_matrix labeled
        G = np.zeros([n_samples, n_samples])

        for i in range(n_samples-1):
            for j in range(i, n_samples):
                G[i, j] = y[i]*y[j]*self.kernel(X[i, :], X[j, :])
                G[j, i] = G[i, j]

        # Projected gradient descent
        loss = []
        for i in range(iterations):
            # Gradient of the dual form with respect to lagrange multipliers.
            dl = np.dot(lagr_multipliers, G) - 1

            # Update lagrange multipliers.
            lagr_multipliers -= (1/n_samples)*learning_rate*dl

            # Project update to satisfy constrains.
            lagr_multipliers = np.clip(lagr_multipliers, 0, self.C)

            # Compute loss.
            loss.append(self.loss(lagr_multipliers, G))

        # Only vectors with multipler greater than 0 contibute
        # to the margin.
        supported = lagr_multipliers > 1e-7
        self.lagr_multipliers = lagr_multipliers[supported]
        self.supported_vectors = X[supported, :]
        self.labels = y[supported]

        # Compute intercept
        bias = []
        for i in range(len(self.labels)):
            if self.lagr_multipliers[i] < self.C:
                b = self.labels[i]
                for j in range(len(self.labels)):
                    if i != j:
                        k = self.kernel(self.supported_vectors[j],
                                        self.supported_vectors[i])
                        b -= self.lagr_multipliers[j] * self.labels[j] *k

                bias.append(b)

        self.b = np.mean(np.array(bias))
        return loss

    def loss(self, lagr_multipliers, G):
        # Return dual form cost function

        target = np.dot(np.dot(lagr_multipliers, G), lagr_multipliers)
        constraint = np.sum(lagr_multipliers)
        return 0.5 * (target - constraint)

    def _predict_single(self, x):
        K = np.array([self.kernel(xi, x) for xi in self.supported_vectors])
        return (np.sum(self.lagr_multipliers * self.labels * K) + self.b) > 0

    def predict(self, X):
        X = np.atleast_2d(X)
        y_hat = np.zeros(X.shape[0])
        for i in range(len(y_hat)):
            y_hat[i] = self._predict_single(X[i, :])

        return y_hat
