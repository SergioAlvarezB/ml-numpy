from __future__ import division

import numpy as np
from collections import namedtuple


Split = namedtuple('Split',
    ['feature_ix',
     'th',
     'X_false',
     'y_false',
     'X_true',
     'y_true'])


class TreeNode:
    """DecisionTree Node. If split node, stores the feature index and the
    value to make the decision, `th`. If represents a leaf then the attribute
    `value` contains a distinct than None value.
    """
    def __init__(self,
                 value=None,
                 feature_ix=None,
                 th=None,
                 false_branch=None,
                 true_branch=None):

        self.value = value
        self.feature_ix = feature_ix
        self.th = th
        self.false_branch = false_branch
        self.true_branch = true_branch


class DecisionTree:
    """Base class for a decision tree. To build a classification or regression
    model the `_impurity_func` and the `_compute_leaf_value` methods must be
    implemented at inheritance.

    Parameters
    ----------
    min_split_samples : `int`, optional
        Minimum number of samples at a node to perform a split. If the number
        of samples is not enough then a leaf value is computed. Defaults to 2.

    max_depth : `int`, optional
        Maximum depth of the decision tree. If this value is reached then a
        leaf value is computed for the node. Defaults to inf.

    min_impurity : `float`, optional
        Minimum improvement in impurity needed to make a split. If a split
        does not reduce impurity by this much then a leaf value is computed.
        Defaults to 1e-7.

    p : `float`, optional
        Proportion of the features to look at each split. Must be in the range
        [0, 1]. Indicates the size of the subset of features, the sepecific
        features are chosen randomly at each split.

    """

    def __init__(self,
                 min_split_samples=2,
                 max_depth=float('inf'),
                 min_impurity=1e-7,
                 p=1.0):

        self.root = None
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.p = p

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 1)

    def predict(self, X):
        return np.array([self._predict_single(X[i, :])
                         for i in range(X.shape[0])])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if n_samples < self.min_split_samples or depth >= self.max_depth:
            value = self._compute_leaf_value(y)
            return TreeNode(value=value)

        impurity_redc = 0

        features_idxs = np.random.permutation(n_features)
        # Feature bagging
        features_idxs = features_idxs[:int(self.p*n_features)]
        for feature_ix in features_idxs:
            for th in np.unique(X[:, feature_ix]):

                split = self._make_split(X, y, feature_ix, th)

                curr_impurity = self._impurity_redc(
                        y,
                        split.y_false,
                        split.y_true)

                if curr_impurity > impurity_redc:
                    impurity_redc = curr_impurity
                    best_split = split

        if impurity_redc >= self.min_impurity:
            true_branch = self._build_tree(
                    best_split.X_true,
                    best_split.y_true,
                    depth+1)
            false_branch = self._build_tree(
                    best_split.X_false,
                    best_split.y_false,
                    depth+1)
            return TreeNode(
                    feature_ix=best_split.feature_ix,
                    th=best_split.th,
                    false_branch=false_branch,
                    true_branch=true_branch)

        # We have reached a leaf
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

        redc = (wfalse*self._impurity_func(y_false)
                + wtrue*self._impurity_func(y_true))

        return self._impurity_func(y) - redc

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

        return Split(
                feature_ix=feature_ix,
                th=th,
                X_false=X_false,
                y_false=y_false,
                X_true=X_true,
                y_true=y_true)

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
    """Implements a decision tree classifier. Inherits from DecisionTree
    implementing `_compute_leaf_value` and `_impurity_func`.
    """
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
    """Implements a decision tree based regression model. Inherits from
    DecisionTree implementing `_compute_leaf_value` and `_impurity_func`.
    """
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
