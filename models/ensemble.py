import numpy as np
import copy
from multiprocessing.dummy import Pool


class Bagging:
    """Bagging base class. Implements common methods to classification and
    regression tasks when bagging. Trains `n_models` copies of `base_model`
    using bootstrapping, each trained on a different random sample of the
    dataset with replacement.

    Parameters
    ----------
    base_model : `object`
        Initialized model to replicate.

    n_models : `int`, optional
        Number of models to train.

    """
    def __init__(self, base_model, n_models=10):
        self.base_model = base_model
        self.n_models = n_models

    def fit(self, X, y):

        self.models = [self._fit_model(copy.deepcopy(self.base_model), X, y)
                       for _ in range(self.n_models)]

    def predict(self, x):
        raise NotImplementedError

    def _fit_model(self, model, X, y):

        (n_samples, n_features) = X.shape

        # Bootstrap
        boots_ix = np.random.choice(n_samples, size=n_samples)
        x_train = X[boots_ix, :]
        y_train = y[boots_ix]

        model.fit(x_train, y_train)

        return model


class BaggingClassifier(Bagging):
    """Inherits from Bagging base class. Implements classification by taking
    the most voted label among all the predictors.
    """

    def predict(self, x):
        n_samples = x.shape[0]
        predictions = np.zeros((n_samples, self.n_models))
        for i in range(self.n_models):
            predictions[:, i] = self.models[i].predict(x)

        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(),
                                          1, predictions)
        return predictions


class BaggingRegressor(Bagging):
    """Inherits from Bagging base class. Implements regression by taking
    the mean output among all the predictors outputs.
    """

    def predict(self, x):
        n_samples = x.shape[0]
        predictions = np.zeros((n_samples, self.n_models))
        for i in range(self.n_models):
            predictions[:, i] = self.models[i].predict(x)

        predictions = np.mean(predictions, axis=1)
        return predictions


class GradientBoostingRegressor:
    """Implements gradient bossting meta-algorithm for regression. At each
    training step trains a new estimator, a copy of `base_model` to predict the
    residuals of the current model the adds the estimator to the current model.

    Parameters
    ----------
    base_model : `object`
        Initialized model to replicate.

    learning_rate : `float`, optional
        Factor by which to multiply the prediction of each new estimator added
        to the model. Defaults to 0.5.

    max_models : `int`, optional
        Maximum number of estimators to add to the model. Defaults to 100.

    min_rmse : `float`, optional
        Threshold of minimum rmse improvement needed to continue training
        models.
    """

    def __init__(self, base_model, learning_rate=0.5, max_models=100,
                 min_rmse=1e-7):
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.max_models = max_models
        self.min_rmse = min_rmse

    def fit(self, X, y):
        self.initial_model = copy.deepcopy(self.base_model)
        self.initial_model.fit(X, y)
        residuals = y-self.initial_model.predict(X)
        self.models = []

        curr_rmse = np.mean(np.square(residuals))
        rmse_improv = float('inf')

        while (rmse_improv >= self.min_rmse
               and len(self.models) < self.max_models):
            # Train new estimator
            new_model = copy.deepcopy(self.base_model)
            new_model.fit(X, residuals)

            self.models.append(new_model)
            residuals = y-self.predict(X)

            new_rmse = np.mean(np.square(residuals))
            rmse_improv = curr_rmse - new_rmse

            curr_rmse = new_rmse

    def predict(self, x):
        predictions = self.initial_model.predict(x)

        for model in self.models:
            predictions += self.learning_rate*model.predict(x)

        return predictions
