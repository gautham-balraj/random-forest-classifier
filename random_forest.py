"""
A Random Forest Classifier implementation in Python.

This module provides a Random Forest Classifier that can be used for classification tasks.
It utilizes the Decision Tree Classifier from the 'decision_tree' module to build an ensemble
of decision trees, and performs bootstrap sampling to train each tree on a different subset
of the training data.

Classes:
    MyRandomForestClassifier: A Random Forest Classifier class.


Usage:
    >>> from random_forest import MyRandomForestClassifier
    >>> from decision_tree import DecisionTree
    >>> X_train, X_test, y_train, y_test = load_data()
    >>> rf = MyRandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2)
    >>> rf.fit(X_train, y_train)
    >>> y_pred = rf.predict(X_test)


"""

from decision_tree import DecisionTree
import numpy as np
from collections import Counter


class MyRandomForestClassifier:
    """
    A Random Forest Classifier class.

    Attributes:
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of each tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        n_features (int): The number of features to consider when looking for the best split.
        trees (list): A list of Decision Tree objects, representing the trees in the forest.

    Methods:
        fit(X, y): Fit the random forest to the training data.
        predict(X): Predict the target values for the given data.
        _bootstrap_sample(X, y): Perform bootstrap sampling on the training data.
        _most_common_label(y): Find the most common label in the target values.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=100,
        min_samples_split=2,
        n_features=None,
        criterion='entropy'
    ):
        """
        Initialize a Random Forest Classifier.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
            max_depth (int, optional): The maximum depth of each tree. Defaults to 100.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 2.
            n_features (int, optional): The number of features to consider when looking for the best split. Defaults to None, which means all features will be considered.
            criterion (str, optional): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. Defaults to "entropy".
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest to the training data.

        Args:
            X (numpy.ndarray): The training data features.
            y (numpy.ndarray): The target values.
        """
        for i in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
                criterion='entropy'
            )
            x_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target values for the given data.

        Args:
            X (numpy.ndarray): The data features.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_predictions])
        return predictions

    def _bootstrap_sample(self, X, y):
        """
        Perform bootstrap sampling on the training data.

        Args:
            X (numpy.ndarray): The training data features.
            y (numpy.ndarray): The target values.

        Returns:
            tuple: The bootstrap sample of features and target values.
        """
        n_samples = X.shape[0]
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[sample_indices], y[sample_indices]

    def _most_common_label(self, y):
        """
        Find the most common label in the target values.

        Args:
            y (numpy.ndarray): The target values.

        Returns:
            any: The most common label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common