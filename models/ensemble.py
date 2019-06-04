import numpy as np
import copy
from multiprocessing.dummy import Pool

from models.dummy import DummyRegressor


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
        # Fit `n_models` independantly.
        self.models = [self._fit_model(copy.deepcopy(self.base_model), X, y)
                       for _ in range(self.n_models)]

    def predict(self, X):
        raise NotImplementedError

    def _fit_model(self, model, X, y):
        n_samples, n_features = X.shape

        # Bootstrap.
        boots_ix = np.random.choice(n_samples, size=n_samples)
        x_train = X[boots_ix, :]
        y_train = y[boots_ix]

        model.fit(x_train, y_train)
        return model


class BaggingClassifier(Bagging):
    """Inherits from Bagging base class. Implements classification by taking
    the most voted label among all the predictors.
    """

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_models))
        for i in range(self.n_models):
            predictions[:, i] = self.models[i].predict(X)

        predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=1,
                arr=predictions.astype(int))

        return predictions


class BaggingRegressor(Bagging):
    """Inherits from Bagging base class. Implements regression by taking
    the mean output among all the predictors outputs.
    """

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_models))
        for i in range(self.n_models):
            predictions[:, i] = self.models[i].predict(X)

        predictions = np.mean(predictions, axis=1)
        return predictions


class GradientBoosting:
    """Implements gradient bossting meta-algorithm.
    At each training step trains a new estimator, a copy of `base_model`,
    to predict the residuals of the current model, in the regression task,
    or a scaled version of the gradient of the logistic function, in the
    case of classification. Then the estimator to the current model.

    Parameters
    ----------
    base_model : `object`
        Initialized model to replicate.

    learning_rate : `float`, optional
        Factor by which to multiply the prediction of each new estimator added
        to the model. Defaults to 0.5.

    max_models : `int`, optional
        Maximum number of estimators to add to the model. Defaults to 100.
    """

    def __init__(self, base_model, learning_rate=0.5, max_models=100):
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.max_models = max_models

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class GradientBoostingRegressor(GradientBoosting):
    """Implements methods to train gradient boosting meta-model
    for regression.

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

    def __init__(self,
                 base_model,
                 learning_rate=0.5,
                 max_models=100,
                 min_rmse=1e-7):
        # Initialize with parent class method.
        super().__init__(base_model, learning_rate, max_models)
        self.min_rmse = min_rmse

    def fit(self, X, y):
        self.initial_model = DummyRegressor(y)
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

    def predict(self, X):
        predictions = self.initial_model.predict(X)

        for model in self.models:
            predictions += self.learning_rate*model.predict(X)

        return predictions


class GradientBoostingClassifier(GradientBoosting):
    """Implements methods to train gradient boosting meta-model
    for classification.

    Parameters
    ----------
    base_model : `object`
        Initialized model to replicate.

    learning_rate : `float`, optional
        Factor by which to multiply the prediction of each
        new estimator added to the model. Defaults to 0.5.

    max_models : `int`, optional
        Maximum number of estimators to add to the model.
        Defaults to 100.

    min_loss : `float`, optional
        Threshold of minimum loss improvement needed to continue training
        models. Otherwise the training is considered to have converged.
    """

    def __init__(self,
                 base_model,
                 learning_rate=0.5,
                 max_models=100,
                 min_loss=1e-5):
        super().__init__(base_model, learning_rate, max_models)
        self.min_loss = min_loss
        self.update_steps = []

    def _loss(self, y, preds):
        return -(y*np.log(preds) + (1-y)*np.log(1-preds))

    def fit(self, X, y):
        self.initial_model = DummyRegressor(y, classification=True)

        for i in range(self.max_models):
            probs = self.predict_proba(X)
            grads = (1-probs)*(y + y-1)

            loss = self._loss(y, probs)

            new_model = copy.deepcopy(self.base_model)
            new_model.fit(grads)

            # Estimate update step
            logits = self.predict_logits(X)
            rho = 0  # Update step
            preds = new_model.predict(X)
            probs = sigmoid(logits + rho*preds)

            prev_loss = loss

            while True:
                descent_dir = (1-probs)*(y + y-1)*preds

                probs = sigmoid(logits + (rho + 0.01*descent_dir)*preds)
                new_loss = self._loss(y, probs)
                if new_loss >= loss:
                    break
                else:
                    rho += 0.01*descent_dir
                    loss = new_loss

            self.models.append(new_model)
            self.update_steps.append(rho)

            if loss + self.min_loss <= prev_loss:
                # Converged
                break

    def predict_logits(self, X):
        predictions = self.initial_model.predict(X)

        for i, model in enumerate(self.models):
            coef = self.learning_rate*self.update_steps[i]
            predictions += coef*model.predict(X)

        return predictions

    def predict_proba(self, X):
        predictions = self.predict_logits(X)
        # Logistic function.
        return sigmoid(-predictions)

    def predict(self, X):
        return np.round(self.predict_proba(X))


def sigmoid(z):
    return 1. / (1. + np.exp(-z))
