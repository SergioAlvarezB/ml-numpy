import numpy as np


class TreeNode:
    def __init__(self, value=None, feature_ix=None, th=None,
                 true_branch=None, false_branch=None):
        self.value = value
        self.feature_ix = feature_ix
        self.th = th
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree:

    def __init__(self, min_split_samples=2, max_depth=float('inf'),
                 min_impurity=1e-7):
        self.root = None
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self._impurity_func = None

    def fit(X, y):
        self.root = _build_tree(X, y)

    def _build_tree(X, y):
        (n_samples, n_features) = X.shape

        if n_samples < self.min_split_samples:
            value = self._compute_leaf_value(y)
            return TreeNode(value=value)

        impurity = 0
        for feature_ix in range(n_features):
            for th in np.unique(X[:, feature_ix]):

                # Make split
                X_drop = np.delete(X, feature_ix, 1)
                true = X[:, feature_ix] >= th
                # False split
                X_false = X_drop[~true, :]
                y_false = y[~true]
                false_fr = len(y_false)/float(n_samples)
                # True split
                X_true = X_drop[true, :]
                y_true = y[true, :]
                true_fr = 1.0 - false_rate

                curr_impurity = false_fr*self._impurity_func(X_false, y_false)\
                    + true_fr*self._impurity_func(X_true, y_true)

                if curr_impurity > impurity:
                    impurity = curr_impurity
                    best_split = {"feature_ix": feature_ix,
                                  "th": th,
                                  "X_false": X_false,
                                  "y_false": y_false,
                                  "X_true": X_true,
                                  "y_true": y_true}

        if impurity >= self.min_impurity:
            true_branch = self._build_tree(best_split.X_true,
                                           best_split.y_true)
            false_branch = self._build_tree(best_split.X_false,
                                            best_split.y_false)
            return TreeNode(feature_ix=best_split.feature_ix,
                            th=best_split.th,
                            true_branch=true_branch,
                            false_branch=false_branch)

        value = self._compute_leaf_value(y)
        return TreeNode(value=value)

    def _predict_single(self, x, node=None):
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        if x[feature_ix] >= node.th:
            return self._predict_single(x, node.true_branch)
        else:
            return self._predict_single(x, node.false_branch)

    def predict(self, X):
