"""
A Decision Tree Classifier implementation from scratch in Python.

This module provides a Decision Tree Classifier that can be used for classification tasks.
It supports the Gini impurity and Information Gain criteria for splitting nodes.

Classes:
    Node: A class representing a node in the decision tree.
    DecisionTree: A Decision Tree Classifier class.

Functions:
    None

Usage:
    >>> from decision_tree import DecisionTree
    >>> X_train, X_test, y_train, y_test = load_data()
    >>> dt = DecisionTree(criterion='gini', max_depth=5, min_samples_split=2)
    >>> dt.fit(X_train, y_train)
    >>> y_pred = dt.predict(X_test)
"""

import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        A Node class representing a node in the decision tree.

        Args:
            feature (int, optional): The feature index used for splitting the node.
            threshold (float, optional): The threshold value for splitting the node.
            left (Node, optional): The left child node.
            right (Node, optional): The right child node.
            value (any, optional): The value of the node if it's a leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the current node is a leaf node.

        Returns:
            bool: True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


class DecisionTree:
    def __init__(self, criterion="entropy", min_samples_split=2, max_depth=100, n_features=None):
        """
        A Decision Tree classifier.

        Args:
            criterion (str, optional): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node.
            max_depth (int, optional): The maximum depth of the tree.
            n_features (int, optional): The number of features to consider when looking for the best split.
        """
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Args:
            X (numpy.ndarray): The training data features.
            y (numpy.ndarray): The target values.
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Args:
            X (numpy.ndarray): The data features.
            y (numpy.ndarray): The target values.
            depth (int, optional): The current depth of the node.

        Returns:
            Node: The root node of the decision tree.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Randomly select features
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Split the data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, x, y, feat_idxs):
        """
        Find the best feature and threshold for splitting the data.

        Args:
            x (numpy.ndarray): The data features.
            y (numpy.ndarray): The target values.
            feat_idxs (list): The list of feature indices to consider.

        Returns:
            tuple: The best feature index and threshold value.
        """
        if self.criterion == "gini":
            return self._best_split_gini(x, y, feat_idxs)
        elif self.criterion == "entropy":
            return self._best_split_entropy(x, y, feat_idxs)
        else:
            raise ValueError("Criterion not supported")

    def _best_split_gini(self, x, y, feat_idxs):
        """
        Find the best feature and threshold for splitting the data using the Gini impurity.

        Args:
            x (numpy.ndarray): The data features.
            y (numpy.ndarray): The target values.
            feat_idxs (list): The list of feature indices to consider.

        Returns:
            tuple: The best feature index and threshold value.
        """
        max_gini_gain = -np.inf
        best_feature, best_threshold = None, None
        for feature in feat_idxs:
            x_column = x[:, feature]
            thresholds = np.unique(x_column)
            for thres in thresholds:
                gini_gain = self._gini_gain(y, x_column, thres)
                if gini_gain > max_gini_gain:
                    max_gini_gain = gini_gain
                    best_feature = feature
                    best_threshold = thres
        return best_feature, best_threshold

    def _best_split_entropy(self, x, y, feat_idxs):
        """
        Find the best feature and threshold for splitting the data using the information gain.

        Args:
            x (numpy.ndarray): The data features.
            y (numpy.ndarray): The target values.
            feat_idxs (list): The list of feature indices to consider.

        Returns:
            tuple: The best feature index and threshold value.
        """
        max_ig = -np.inf
        best_feature, best_threshold = None, None
        for feature in feat_idxs:
            x_column = x[:, feature]
            thresholds = np.unique(x_column)
            for thres in thresholds:
                ig = self._information_gain(y, x_column, thres)
                if ig > max_ig:
                    max_ig = ig
                    best_feature = feature
                    best_threshold = thres
        return best_feature, best_threshold

    def _most_common_label(self, y):
        """
        Find the most common label in the target values.

        Args:
            y (numpy.ndarray): The target values.

        Returns:
            any: The most common label.
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _split(self, X_column, split_thresh):
        """
        Split the data based on the given threshold.

        Args:
            X_column (numpy.ndarray): The feature column.
            split_thresh (float): The threshold value.

        Returns:
            tuple: The indices of the left and right splits.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        """
        Calculate the Gini impurity for the target values.

        Args:
            y (numpy.ndarray): The target values.

        Returns:
            float: The Gini impurity.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _gini_gain(self, y, x_column, threshold):
        """
        Calculate the Gini gain for the given feature and threshold.

        Args:
            y (numpy.ndarray): The target values.
            x_column (numpy.ndarray): The feature column.
            threshold (float): The threshold value.

        Returns:
            float: The Gini gain.
        """
        parent_gini = self._gini(y)

        left_idxs, right_idxs = self._split(x_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        left_gini = self._gini(y[left_idxs])
        right_gini = self._gini(y[right_idxs])

        gini_gain = parent_gini - (n_l / n) * left_gini - (n_r / n) * right_gini
        return gini_gain

    def _information_gain(self, y, x_column, threshold):
        """
        Calculate the information gain for the given feature and threshold.

        Args:
            y (numpy.ndarray): The target values.
            x_column (numpy.ndarray): The feature column.
            threshold (float): The threshold value.

        Returns:
            float: The information gain.
        """
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(x_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])

        ig = parent_entropy - (n_l / n) * left_entropy - (n_r / n) * right_entropy
        return ig

    def _entropy(self, y):
        """
        Calculate the entropy for the target values.

        Args:
            y (numpy.ndarray): The target values.

        Returns:
            float: The entropy.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def predict(self, X):
        """
        Predict the target values for the given data.

        Args:
            X (numpy.ndarray): The data features.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to predict the target value for the given data point.

        Args:
            x (numpy.ndarray): The data point.
            node (Node): The current node.

        Returns:
            any: The predicted target value.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)