import numpy as np
# TODO: Comment, add radial kernel


def linear(x, y):
    return np.dot(x, y)


def polynomial(x, y, d=2):
    return np.power(np.dot(x, y)+1, d)
