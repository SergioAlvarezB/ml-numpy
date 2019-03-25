"""Collection of toy datasets.

This modules provides an interface to load some toy datasets with the
purpose of making notebooks and example scripts more readable.
"""

import numpy as np
from sklearn.datasets import make_blobs


def blobs_classification_dataset(features, classes, samples=100):
    """Generates a vanilla classification dataset.

    A basic wrapper for the make_blobs dataset generator of sklearn:
    https://scikit-learn.org/stable/modules/generated

    Parameters
    ----------

    features : `int`
        Number of dimensions of the input data.

    classes : `int`
        Size of the set of classes, e.g. classes=2 implies
        binary classification.

    samples : `int`, optional
        Total number of samples to generate.

    """

    x, y = make_blobs(n_samples=samples,
                      n_features=features,
                      centers=classes)

    # Split the data into training and test set
    perm = np.random.permutation(samples)
    x = x[perm]
    y = y[perm]

    ix = int(samples*0.8)

    x_train, y_train = x[:ix], y[:ix]
    x_test, y_test = x[ix:], y[ix:]

    return (x_train, y_train), (x_test, y_test)


def radial_classification_dataset(classes=2, samples=100, r_inc=1, noise=0.1):
    # TODO: Comment

    radius = np.arange(classes)/float(r_inc)

    samples_pclass = samples//classes
    samples = samples_pclass*classes
    x = np.zeros([samples, 2])
    y = np.zeros(samples)
    for i, r in enumerate(radius):
        # Simulate data in polar coordinates
        e = np.random.randn(samples_pclass, 2)*noise
        theta = np.random.rand(samples_pclass)*2*np.pi
        curr_x = np.array([r*np.cos(theta), r*np.sin(theta)]).T + e
        x[i*samples_pclass:(i+1)*samples_pclass, :] = curr_x
        y[i*samples_pclass:(i+1)*samples_pclass] = i

    # Split the data into training and test set
    perm = np.random.permutation(samples)
    x = x[perm]
    y = y[perm]

    ix = int(samples*0.8)

    x_train, y_train = x[:ix], y[:ix]
    x_test, y_test = x[ix:], y[ix:]

    return (x_train, y_train), (x_test, y_test)
