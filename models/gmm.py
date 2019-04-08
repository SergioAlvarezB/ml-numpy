import numpy as np
from k_means import KMeans

class GMM:
    """Implementes Gaussian Mixture Model fitted with EM algorithm. Similar to
    K-means, iteratively adjusts the clusters to match better the points
    belonging to them and then recompute which points belong to what cluster.
    In this case the clusters are soft, each point has a likelihood for each
    possible clusters. Clusters are modeled ase multivariate gaussians. The
    model is fitted using the EM algorithm.
    """

    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = 100
        self.weights = None


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


    def _expectation_step(self, X):
        (n_samples, n_features) = X.shape
        # Computes likelihood for each sample of each gaussian
        h = np.zeros(n_samples, self.k)
        for k in self.k:
            h[:, k] = self._multinomial_gaussian(X,
                                                 self.means[k],
                                                 self.variances[k])
            h[:, k] *= self.weights[k]

        h = h/np.sum(h, axis=1)

        return h

    def _maximization_step(self, X, h):
        (n_samples, n_features) = X.shape
        # Update weights
        self.weights = (1./n_samples)*np.sum(h, axis=0)

        # Update gaussians
        for k in self.k:
            h_sum = np.sum(h[:, k])
            self.means[k] = np.sum(h[:, k]*X)/h_sum
            # TODO: compute properly variance
            diff = (X-self.means[k])
            self.variances = np.matmul(diff.T, diff)/h_sum


    def _multinomial_gaussian(self, x, mean, variance):
        factor = 1./np.sqrt(np.linalg.det(variance)*(2*np.pi)**self.k)
        diff = x-mean
        inv_var = np.linalg.inv(variance)
        exp = np.exp(-0.5*np.sum((diff @ inv_var)* diff), axis=1)

        return factor*exp


    def _init_gaussians(self, X):

        # Init gaussians using the result of running kmeans
        self.kmodel = KMeans(k=self.k)
        self.kmodel.fit(X)

        self.means = self.kmodel._centroids
        self.variances = self.kmodel._variances

        self.weights = [1./self.k]*self.k
