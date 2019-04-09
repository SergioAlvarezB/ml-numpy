import numpy as np
from collections import Counter


class KMeans:
    """Implements K-Means. Clustering that fits k clusters by iteratively
    assigning points to the cluster represented by the nearest centroid and
    moving the centroid to the center of the cluster.

    Parameters
    ----------
    k : `int`, optional
        Number of clusters. Defaults to 2.

    max_iter : `int`, optional
        Max iterations to perform before the algorithm stops training if
        convergence is not achieved.
    """

    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X, re_init=True, verbose=False):
        """Fits `self.k` clusters to the data `X`.

        Parameters
        ----------
        X : `numpy.ndarray` (n_examples, n_features)
            Training data.

        re_init : `boolean`, optional
            Whether to reinitialize centroids. Defaults to True. If this is the
            first call to fit, centroids will be initialized eitherway.
        """

        # Initialize centroids
        if re_init or self.centroids is None:
            self._init_centroids(X)

        for i in range(self.max_iter):
            clusters = self._clusterize(X)
            new_centroids = self._compute_centroids(X, clusters)

            # Convergence
            if (new_centroids == self.centroids).all():
                if verbose:
                    print("Algorithm converged at iteration nº{}".format(i))
                break

            self.centroids = new_centroids

    def predict(self, x, score=False):
        """Assign each sample to a cluster.

        Parameters
        ----------
        x : `numpy.ndarray` (n_samples, n_features)
            New data to predict

        Returns
        -------
        clusters : `numpy.array` (n_samples,)
            A label for each sample in `x` indicating the assigned cluster.
        """
        if self.centroids is None:
            raise ValueError('The model has not been fitted to any data yet!')

        return self._clusterize(x, score)

    def _init_centroids(self, X):
        # Randomly choose k samples from the data as centroids
        (n_samples, n_features) = X.shape

        centroids_ix = np.random.choice(n_samples, size=self.k, replace=False)

        self.centroids = X[centroids_ix, :]

    def _clusterize(self, X, score=False):
        # Assign a cluster to each sample, computes score as the sum of the
        # inner sum of squares of each cluster
        (n_samples, n_fetures) = X.shape
        dists = np.zeros((n_samples, self.k))
        for k in range(self.k):
            dists[:, k] = np.sum(np.square(X-self.centroids[k, :]), axis=1)

        clusters = np.argmin(dists, axis=1)

        if score:
            score = sum([1.0/(2*n)*np.sum(dists[clusters == c, c])
                         for c, n in Counter(clusters).most_common()])
            return clusters, score

        return clusters

    def _compute_centroids(self, X, clusters):
        # Compute each new centroid as the mean of its cluster
        (n_samples, n_features) = X.shape
        centroids = np.zeros((self.k, n_features))
        for k in range(self.k):
            cluster = X[clusters == k, :]
            centroids[k] = np.mean(cluster, axis=0)

        return centroids

    def _inner_variances(self, X, clusters):
        covariances = []
        for k in range(self.k):
            diff = X[clusters == k, :] - self.centroids[k]
            covariances.append(np.matmul(diff.T, diff))

        return covariances

    def _init_gmm(self, X):
        # Returns centroids and covariance matrices
        self.fit(X)
        clusters = self.predict(X)

        self.covariances = self._inner_variances(X, clusters)

        return self.centroids, self.covariances
