import numpy as np


def linear(x, y):
    """Implements linear kernel, equivalent to inner product"""
    return np.dot(x, y)


def polynomial(x, y, d=2):
    """Implements polynomial kernel.

    Computes the function (x·y + 1)**d, where x·y is the inner product between
    vectors x and y.
    """
    return np.power(np.dot(x, y)+1, d)


def rbf(x, y, gamma=1):
    """Radial basis function.

    Computes the function exp(-gamma*||x-y||**2).
    """
    return np.exp(-gamma*np.sum(np.square(x-y)))
