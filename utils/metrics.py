import numpy as np
import itertools
import copy


def prediction_strength(model, X, x_test=None, test_split=0.2):
    """Computes the prediction strength of a clustering algorithm.
    """

    # If no test set is provided, split the data.
    if x_test is None:
        perm = np.random.permutation(X.shape[0])
        pivot = int(X.shape[0]*test_split)
        x_test, X = X[:pivot], X[pivot:]

    # Create and fit models
    train_model = copy.deepcopy(model)
    test_model = copy.deepcopy(model)

    train_model.fit(X)
    test_model.fit(x_test)

    # Comembership matrix
    y_hat = train_model.predict(x_test)
    if y_hat.ndim > 1:
        y_hat = np.argmax(y_hat, axis=1)

    D = np.zeros((x_test.shape[0], x_test.shape[0]))
    for i in range(x_test.shape[0]):
        for j in range(i, x_test.shape[0]):
            D[i, j] = D[j, i] = y_hat[i] == y_hat[j]

    # prediction strength
    total_score = float('inf')
    y = test_model.predict(x_test)
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    clusters, counts = np.unique(y, return_counts=True)
    for k, n in zip(clusters, counts):
        if n < 2:
            # Cluster isn't big enough
            continue

        # Index of elements in this cluster
        ixs = y == k
        curr_score = np.sum(D[np.ix_(ixs, ixs)])
        curr_score /= (n*(n-1.))
        total_score = min(total_score, curr_score)

    return total_score
