from __future__ import division

import numpy as np
from collections import namedtuple


Split = namedtuple('Split', ['feature_ix', 'th', 'X_false', 'y_false',
                             'X_true', 'y_true'])


class TreeNode:
    def __init__(self, value=None, feature_ix=None, th=None,
                 false_branch=None, true_branch=None):
        self.value = value
        self.feature_ix = feature_ix
        self.th = th
        self.false_branch = false_branch
        self.true_branch = true_branch


class DecisionTree:

    def __init__(self, min_split_samples=2, max_depth=float('inf'),
                 min_impurity=1e-7):
        self.root = None
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth
        self.min_impurity = min_impurity

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 1)

    def predict(self, X):
        return np.array([self._predict_single(X[i, :])
                         for i in range(X.shape[0])])

    def _build_tree(self, X, y, depth):
        (n_samples, n_features) = X.shape

        if n_samples < self.min_split_samples or depth >= self.max_depth:
            value = self._compute_leaf_value(y)
            return TreeNode(value=value)

        impurity_redc = 0
        for feature_ix in range(n_features):
            for th in np.unique(X[:, feature_ix]):

                split = self._make_split(X, y, feature_ix, th)

                curr_impurity = self._impurity_redc(y,
                                                    split.y_false,
                                                    split.y_true)

                if curr_impurity > impurity_redc:
                    impurity_redc = curr_impurity
                    best_split = split

        if impurity_redc >= self.min_impurity:
            true_branch = self._build_tree(best_split.X_true,
                                           best_split.y_true,
                                           depth+1)
            false_branch = self._build_tree(best_split.X_false,
                                            best_split.y_false,
                                            depth+1)
            return TreeNode(feature_ix=best_split.feature_ix,
                            th=best_split.th,
                            false_branch=false_branch,
                            true_branch=true_branch)

        value = self._compute_leaf_value(y)
        return TreeNode(value=value)

    def _compute_leaf_value(self, y):
        raise NotImplementedError

    def _impurity_func(self, y):
        raise NotImplementedError

    def _impurity_redc(self, y, y_false, y_true):
        # Computes the improvement in impurity
        wfalse = len(y_false)/len(y)
        wtrue = 1.0 - wfalse

        return self._impurity_func(y) - (wfalse*self._impurity_func(y_false)
                                         + wtrue*self._impurity_func(y_true))

    def _make_split(self, X, y, feature_ix, th):
        # Split data according to value `th` of feature `feature_ix`

        # Feature is numeric
        if isinstance(th, (int, float)):
            true_mask = X[:, feature_ix] >= th
        else:
            true_mask = X[:, feature_ix] == th

        # False split
        X_false = X[~true_mask, :]
        y_false = y[~true_mask]

        # True split
        X_true = X[true_mask, :]
        y_true = y[true_mask]

        return Split(feature_ix, th, X_false, y_false, X_true, y_true)

    def _predict_single(self, x, node=None):
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        # Feature is numeric
        if isinstance(node.th, (int, float)):
            if x[node.feature_ix] >= node.th:
                return self._predict_single(x, node.true_branch)
            else:
                return self._predict_single(x, node.false_branch)
        # Feature is categorical
        else:
            if x[node.feature_ix] == node.th:
                return self._predict_single(x, node.true_branch)
            else:
                return self._predict_single(x, node.false_branch)


class DecisionTreeClassifier(DecisionTree):
    def _compute_leaf_value(self, y):
        # Return the most apearing label
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _impurity_func(self, y):
        # Computes entropy as impurity
        if isinstance(y, (int, float)):
            return 0
        N = len(y)
        _, counts = np.unique(y, return_counts=True)
        probs = counts/N
        entropy = -np.sum(probs*np.log(probs))

        return entropy


class DecisionTreeRegressor(DecisionTree):
    def _compute_leaf_value(self, y):
        # Return the mean value
        return np.mean(y)

    def _impurity_func(self, y):
        # Computes mean square error as impurity
        N = len(y)
        if N == 0:
            return 0
        ym = np.mean(y)
        mse = 1/N*np.sum(np.square(y-ym))

        return mse
