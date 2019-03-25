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
    # TODO: Comment all class, add radial kernel
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
        G = np.zeros([n_samples, n_samples])
        for i in range(n_samples-1):
            for j in range(i, n_samples):
                G[i, j] = y[i]*y[j]*self.kernel(X[i, :], X[j, :])
                G[j, i] = G[i, j]

        # Projected gradient descent
        loss = []
        for i in range(iterations):
            # Gradient of the dual form with respect to lagrange multipliers
            dl = np.dot(lagr_multipliers, G) - 1

            # Update lagrange multipliers
            lagr_multipliers -= (1/n_samples)*learning_rate*dl

            # Project update to satisfy constrains
            lagr_multipliers = np.clip(lagr_multipliers, 0, self.C)

            # Compute loss
            loss.append(self.loss(lagr_multipliers, G))

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
                        b -= (self.lagr_multipliers[j] * self.labels[j]
                              * self.kernel(self.supported_vectors[j],
                                            self.supported_vectors[i]))
                bias.append(b)
        self.b = np.mean(np.array(bias))

        return loss

    def loss(self, lagr_multipliers, G):
        # Return dual form cost function
        return 0.5*(np.dot(np.dot(lagr_multipliers, G), lagr_multipliers)
                    - np.sum(lagr_multipliers))

    def _predict_single(self, x):
        K = np.array([self.kernel(xi, x) for xi in self.supported_vectors])
        return (np.sum(self.lagr_multipliers*self.labels*K) + self.b) > 0

    def predict(self, x):
        x = np.atleast_2d(x)
        y_hat = np.zeros(x.shape[0])
        for i in range(len(y_hat)):
            y_hat[i] = self._predict_single(x[i, :])

        return y_hat
