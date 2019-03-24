import numpy as np
from utils import kernels


class SVM:
    # TODO: Comment all class, add radial kernel, encapsulate loss in a method
    def __init__(self, kernel='linear', C=1, degree=2):
        self.C = C
        self.kernel = self._get_kernel(kernel, degree)

    def _get_kernel(self, kernel, degree):
        # return kernel as a Callable
        if kernel == 'linear':
            return kernels.linear

        if kernel == 'quadratic':
            return lambda x, y: kernels.polynomial(x, y, d=2)

        if kernel == 'polynomial':
            return lambda x, y: kernels.polynomial(x, y, d=degree)

    def fit(self, X, y, iterations=1000, learning_rate=0.01):

        # Transform labels from format {0, 1} to {-1, 1}
        y[y == 0] = -1

        (n_samples, n_dimensions) = X.shape

        # Initialize lagrange multipliers
        lagr_multipliers = np.random.rand(n_samples)/n_samples

        # Compute kernel_matrix labeled
        self.G = np.zeros([n_samples, n_samples])
        for i in range(n_samples-1):
            for j in range(i, n_samples):
                self.G[i, j] = y[i]*y[j]*self.kernel(X[i, :], X[j, :])
                self.G[j, i] = self.G[i, j]

        # Projected gradient descent
        loss = []
        for i in range(iterations):
            # Gradient of hinge loss with respect to lagrange multipliers
            dl = 0.5*np.dot(lagr_multipliers, self.G) - 1

            # Update lagrange multipliers
            lagr_multipliers -= (1/n_samples)*learning_rate*dl

            # Project update to satisfy constrains
            lagr_multipliers = np.clip(lagr_multipliers, 0, self.C)

            # Compute loss
            loss.append(0.5*np.dot(np.dot(lagr_multipliers, self.G),
                                   lagr_multipliers)
                        - np.sum(lagr_multipliers))

        supported = lagr_multipliers > 1e-7
        self.lagr_multipliers = lagr_multipliers[supported]
        self.supported_vectors = X[supported, :]
        self.labels = y[supported]

        # Compute intercept
        self.b = self.labels[0]
        for i in range(1, len(self.labels)):
            self.b -= self.lagr_multipliers[i] * self.labels[i]\
                * self.kernel(self.supported_vectors[i],
                              self.supported_vectors[0])

        return loss

    def _predict_single(self, x):
        K = np.array([self.kernel(xi, x) for xi in self.supported_vectors])
        return np.clip(np.sign(np.sum(self.lagr_multipliers*self.labels*K)
                               + self.b), 0, 1)

    def predict(self, x):
        x = np.atleast_2d(x)
        y_hat = np.zeros(x.shape[0])
        for i in range(len(y_hat)):
            y_hat[i] = self._predict_single(x[i, :])

        return y_hat
