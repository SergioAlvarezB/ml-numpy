import numpy as np
from models.k_means import KMeans


class GMM:
    """Implementes Gaussian Mixture Model fitted with EM algorithm. Similar to
    K-means, iteratively adjusts the clusters to match better the points
    belonging to them and then recompute which points belong to what cluster.
    In this case the clusters are soft, each point has a likelihood for each
    possible clusters. Clusters are modeled ase multivariate gaussians. The
    model is fitted using the EM algorithm.

    Parameters
    ----------
    k : `int`, optional
        Number of clusters. Defaults to 2.

    max_iter : `int`, optional
        Max iterations to perform before the algorithm stops training if
        convergence is not achieved.

    th : `float`, optional
        Minimum change in the posterior probabilities `h` to continue training.
        When the change is smaller the training is stopped as the algorithm is
        considered to have converged. Defaylts to 1e-7.
    """

    def __init__(self, k=2, max_iter=100, th=1e-7):
        self.k = k
        self.max_iter = max_iter
        self.weights = None
        self.th = 1e-7

    def fit(self, X, re_init=True, verbose=False):
        """Fits `self.k` multivariate gaussians to `X` using EM algorithm.

        Parameters
        ----------
        X : `numpy.ndarray` (n_examples, n_features)
            Training data.

        re_init : `boolean`, optional
            Whether to reinitialize centroids. Defaults to True. If this is the
            first call to fit, centroids will be initialized eitherway.
        """

        # Initialize gaussians
        if re_init or self.weights is None:
            self._init_gaussians(X)

        prev_h = None
        for i in range(self.max_iter):
            # Expectation
            h = self._expectation_step(X)

            # Maximization
            self._maximization_step(X, h)
            if prev_h is not None and np.linalg.norm(h-prev_h) < self.th:
                # Converged
                break
            prev_h = h

    def predict(self, x):
        return self._expectation_step(x)

    def _expectation_step(self, X):
        (n_samples, n_features) = X.shape
        # Computes likelihood for each sample of each gaussian
        h = np.zeros((n_samples, self.k))
        for k in range(self.k):
            h[:, k] = self._multinomial_gaussian(X,
                                                 self.means[k],
                                                 self.covariances[k])
            h[:, k] *= self.weights[k]

        h = h/np.sum(h, axis=1, keepdims=True)

        return h

    def _maximization_step(self, X, h):
        (n_samples, n_features) = X.shape
        # Update weights
        self.weights = (1./n_samples)*np.sum(h, axis=0)

        # Update gaussians
        for k in range(self.k):
            h_sum = np.sum(h[:, k])
            self.means[k] = np.sum(h[:, k, None]*X, axis=0)/h_sum
            # TODO: compute properly variance
            diff = (X-self.means[k])
            self.covariances[k] = np.matmul(diff.T, h[:, k, None]*diff)/h_sum

    def _multinomial_gaussian(self, x, mean, variance):
        factor = 1./np.sqrt(np.linalg.det(variance)*(2*np.pi)**self.k)
        diff = x-mean
        inv_var = np.linalg.pinv(variance)
        exp = np.exp(-0.5*np.sum((diff @ inv_var) * diff, axis=1))

        return factor*exp

    def _init_gaussians(self, X):

        # Init gaussians using the result of running kmeans
        self.means, self.covariances = KMeans(k=self.k)._init_gmm(X)

        self.weights = [1./self.k]*self.k
