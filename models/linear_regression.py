import numpy as np


class LinearRegression:
    """Linear Regression model. Models the relationship between some data and a
    response assuming linear relationship and using ordinary least squares to
    estimate the parameters.

    Parameters
    ----------
    dims : `int`, optional
        Number of features of the training data, without taking into
        account the intercept. Defaults to None, leaving the model
        unconstrained with respect to the number of features until the fit
        method is called for the first time)
    """
    def __init__(self, dims=None):
        self.dims = dims
        self.w = None
        self.r2 = None
        self.std_err = None

    def fit(self, x, y):
        """Fits x data using ordinary least squares method.

        Parameters
        ----------
        x : `numpy.ndarray` (n_examples, n_features)
            Independant variables of the training examples from which to
            estimate the parameters of the model.

        y: `numpy.ndarray` (n_examples, n_features)
            The response corresponding to each training example in ``x``.

        Returns
        -------
        weights : `numpy.ndarray` (n_features+1,)
            The estimated of the model parameters given ``x`` and ``y``

        """

        # If dims setted check for compatibility, if not initialize dims
        if self.dims is not None:
            assert x.shape[1] == self.dims, "Number of dimensions mismatch."
        else:
            self.dims = x.shape[1]

        # Add a first dimension of ones corresponding to the intercept
        x = np.hstack((np.ones([x.shape[0], 1]), x))
        pseudo_inv = np.matmul(np.linalg.pinv(np.matmul(x.T, x)), x.T)
        self.w = np.matmul(pseudo_inv, y)

        # Compute fitness metrics
        self._compute_r2(x, y)
        self._estimate_standard_error(x, y)

        return self.w

    def _compute_r2(self, x, y):
        # Computes r2 metric.
        y_pred = np.matmul(x, self.w)
        y_mean = np.mean(y)

        y_stdev = (y-y_mean)**2
        y_pstdev = (y_pred-y_mean)**2
        self.r2 = np.sum(y_pstdev)/np.sum(y_stdev)

    def _estimate_standard_error(self, x, y):
        # Estimates the standar error of the data.
        y_pred = np.matmul(x, self.w)
        self.std_err = np.sqrt(np.sum((y_pred-y)**2)/(x.shape[0]-2))

    def predict(self, x):
        """Predicts the response using the current estimands.

        Parameters
        ----------
        x : `numpy.ndarray` (n_examples, n_features)
            Data from where to infer a prediction

        Returns
        -------
        predicted : `numpy.ndarray` (n_examples,)
            Predicted values using the last computed weights
        """

        assert self.w is not None, "Unfitted predictor! Call the fit method."

        x = np.atleast_2d(x)
        assert x.shape[1] == self.dims, "Number of dimensions mismatch."

        # Add a first dimension of ones corresponding to the intercept
        x = np.hstack((np.ones([x.shape[0], 1]), x))

        return np.matmul(x, self.w)
