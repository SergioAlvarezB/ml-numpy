import numpy as np


class DummyRegressor:
    """Implements dummy model that always predict as response
    the average of the labels in the training set.

    Parameters
    ----------
    Y : `numpy.array` (n_examples,)
        Targets of the training set. No features are needed to
        fit the model

    classification : `boolean`
        Indicates wether the dummy model will be used for Gradient
        Boosting classification. In this case we predict the odds

    """

    def __init__(self, Y, classification=False):
        self.output = np.mean(Y)
        if classification:
            # ML estimate for logistic function.
            self.output = self.output / (1-self.output)


    def predict(self, X):
        # return the output for every sample queried.
        n_samples = X.shape[0]

        return np.ones(n_samples)*self.output
