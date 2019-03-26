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

    Returns
    -------
    x_train : `numpy.ndarray` (0.8*samples, features)
        Generated points belonging to the training set.

    y_train : `numpy.ndarray` (0.8*samples,)
        Labels corresponding to points in `x_train`.

    x_test : `numpy.ndarray` (0.2*samples, features)
        Generated points belonging to the test set.

    y_train : `numpy.ndarray` (0.2*samples,)
        Labels corresponding to points in `x_test`.

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
    """Generates a 2D classification dataset.

    Data is distributed along 0-centered circuferences. Each class falls within
    a different radius from the center. The radius is increased with the
    category number.

    Parameters
    ----------
    classes : `int`, optional
        Number of unique classes. Defaults to 2.

    samples : `int`, optional
        Total number of samples to generate.

    r_inc : `float`, optional
        Specifies the increment of radius between consecutive classes.

    noise : `float`, optional
        Standard deviation of the noise added to the samples.

    Returns
    -------
    x_train : `numpy.ndarray` (0.8*samples, features)
        Generated points belonging to the training set.

    y_train : `numpy.ndarray` (0.8*samples,)
        Labels corresponding to points in `x_train`.

    x_test : `numpy.ndarray` (0.2*samples, features)
        Generated points belonging to the test set.

    y_train : `numpy.ndarray` (0.2*samples,)
        Labels corresponding to points in `x_test`.

    """

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


def spiral_classification_dataset(classes=2, samples=100, r_inc=1, noise=0.1,
                                  radius=2):
    """Generates a 2D classification dataset.

    Data is distributed in the form a spiral starting at 0. Each class describe
    its own curve. Curves are envenly separated starting with different angular
    offsets.

    Parameters
    ----------
    classes : `int`, optional
        Number of unique classes. Defaults to 2.

    samples : `int`, optional
        Total number of samples to generate.

    r_inc : `float`, optional
        Specifies the increment of radius between consecutive classes.

    noise : `float`, optional
        Standard deviation of the noise added to the samples.

    Returns
    -------
    x_train : `numpy.ndarray` (0.8*samples, features)
        Generated points belonging to the training set.

    y_train : `numpy.ndarray` (0.8*samples,)
        Labels corresponding to points in `x_train`.

    x_test : `numpy.ndarray` (0.2*samples, features)
        Generated points belonging to the test set.

    y_train : `numpy.ndarray` (0.2*samples,)
        Labels corresponding to points in `x_test`.

    """

    # Angular difference between spirals
    offsets = np.arange(classes)/classes * 2*np.pi
    samples_pclass = samples//classes
    revs = float(radius)/r_inc
    r = np.arange(samples_pclass)/float(samples_pclass)*radius
    theta = np.arange(samples_pclass)/float(samples_pclass)*2*np.pi*revs

    # Total samples will be a multiple of number of classes so each class
    # is evenly sampled
    samples = samples_pclass*classes
    x = np.zeros([samples, 2])
    y = np.zeros(samples)
    for i in range(classes):
        # Simulate data in polar coordinates
        e = np.random.randn(samples_pclass, 2)*noise*np.array([r, r]).T
        curr_x = np.array([r*np.cos(theta + offsets[i]),
                           r*np.sin(theta + offsets[i])]).T + e
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
