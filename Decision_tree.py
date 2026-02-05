#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from mlxtend.plotting import plot_decision_regions
from copy import deepcopy
from pprint import pprint


class DecisionTreeCART:

    def __init__(self, max_depth=100, min_samples=2, ccp_alpha=0.0, regression=False):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.ccp_alpha = ccp_alpha
        self.regression = regression
        self.tree = None
        self._y_type = None
        self._num_all_samples = None

    def _set_df_type(self, X, y, dtype):
        X = X.astype(dtype)
        y = y.astype(dtype) if self.regression else y
        self._y_dtype = y.dtype

        return X, y

    @staticmethod
    def _purity(y):
        unique_classes = np.unique(y)

        return unique_classes.size == 1

    @staticmethod
    def _is_leaf_node(node):
        return not isinstance(node, dict)   # if a node/tree is a leaf

    def _leaf_node(self, y):
        class_index = 0

        return np.mean(y) if self.regression else y.mode()[class_index]

    def _split_df(self, X, y, feature, threshold):
        feature_values = X[feature]
        left_indexes = X[feature_values <= threshold].index
        right_indexes = X[feature_values > threshold].index
        sizes = np.array([left_indexes.size, right_indexes.size])

        return self._leaf_node(y) if any(sizes == 0) else left_indexes, right_indexes

    @staticmethod
    def _gini_impurity(y):
        _, counts_classes = np.unique(y, return_counts=True)
        squared_probabilities = np.square(counts_classes / y.size)
        gini_impurity = 1 - sum(squared_probabilities)

        return gini_impurity

    @staticmethod
    def _mse(y):
        mse = np.mean((y - y.mean()) ** 2)

        return mse

    @staticmethod
    def _cost_function(left_df, right_df, method):
        total_df_size = left_df.size + right_df.size
        p_left_df = left_df.size / total_df_size
        p_right_df = right_df.size / total_df_size
        J_left = method(left_df)
        J_right = method(right_df)
        J = p_left_df*J_left + p_right_df*J_right

        return J  # weighted Gini impurity or weighted mse (depends on a method)

    def _node_error_rate(self, y, method):
        if self._num_all_samples is None:
            self._num_all_samples = y.size   # num samples of all dataframe
        current_num_samples = y.size

        return current_num_samples / self._num_all_samples * method(y)

    def _best_split(self, X, y):
        features = X.columns
        min_cost_function = np.inf
        best_feature, best_threshold = None, None
        method = self._mse if self.regression else self._gini_impurity

        for feature in features:
            unique_feature_values = np.unique(X[feature])

            for i in range(1, len(unique_feature_values)):
                current_value = unique_feature_values[i]
                previous_value = unique_feature_values[i-1]
                threshold = (current_value + previous_value) / 2
                left_indexes, right_indexes = self._split_df(X, y, feature, threshold)
                left_labels, right_labels = y.loc[left_indexes], y.loc[right_indexes]
                current_J = self._cost_function(left_labels, right_labels, method)

                if current_J <= min_cost_function:
                    min_cost_function = current_J
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _stopping_conditions(self, y, depth, n_samples):
        return self._purity(y), depth == self.max_depth, n_samples < self.min_samples

    def _grow_tree(self, X, y, depth=0):
        current_num_samples = y.size
        X, y = self._set_df_type(X, y, np.float128)
        method = self._mse if self.regression else self._gini_impurity

        if any(self._stopping_conditions(y, depth, current_num_samples)):
            RTi = self._node_error_rate(y, method)   # leaf node error rate
            leaf_node = f'{self._leaf_node(y)} | error_rate {RTi}'
            return leaf_node

        Rt = self._node_error_rate(y, method)   # decision node error rate
        best_feature, best_threshold = self._best_split(X, y)
        decision_node = f'{best_feature} <= {best_threshold} | ' \
                        f'as_leaf {self._leaf_node(y)} error_rate {Rt}'

        left_indexes, right_indexes = self._split_df(X, y, best_feature, best_threshold)
        left_X, right_X = X.loc[left_indexes], X.loc[right_indexes]
        left_labels, right_labels = y.loc[left_indexes], y.loc[right_indexes]

        # recursive part
        tree = {decision_node: []}
        left_subtree = self._grow_tree(left_X, left_labels, depth+1)
        right_subtree = self._grow_tree(right_X, right_labels, depth+1)

        if left_subtree == right_subtree:
            tree = left_subtree
        else:
            tree[decision_node].extend([left_subtree, right_subtree])

        return tree

    def _tree_error_rate_info(self, tree, error_rates_list):
        if self._is_leaf_node(tree):
            *_, leaf_error_rate = tree.split()
            error_rates_list.append(np.float128(leaf_error_rate))
        else:
            decision_node = next(iter(tree))
            left_subtree, right_subtree = tree[decision_node]
            self._tree_error_rate_info(left_subtree, error_rates_list)
            self._tree_error_rate_info(right_subtree, error_rates_list)

        RT = sum(error_rates_list)   # total leaf error rate of a tree
        num_leaf_nodes = len(error_rates_list)

        return RT, num_leaf_nodes

    @staticmethod
    def _ccp_alpha_eff(decision_node_Rt, leaf_nodes_RTt, num_leafs):

        return (decision_node_Rt - leaf_nodes_RTt) / (num_leafs - 1)

    def _find_weakest_node(self, tree, weakest_node_info):
        if self._is_leaf_node(tree):
            return tree

        decision_node = next(iter(tree))
        left_subtree, right_subtree = tree[decision_node]
        *_, decision_node_error_rate = decision_node.split()

        Rt = np.float128(decision_node_error_rate)
        RTt, num_leaf_nodes = self._tree_error_rate_info(tree, [])
        ccp_alpha = self._ccp_alpha_eff(Rt, RTt, num_leaf_nodes)
        decision_node_index, min_ccp_alpha_index = 0, 1

        if ccp_alpha <= weakest_node_info[min_ccp_alpha_index]:
            weakest_node_info[decision_node_index] = decision_node
            weakest_node_info[min_ccp_alpha_index] = ccp_alpha

        self._find_weakest_node(left_subtree, weakest_node_info)
        self._find_weakest_node(right_subtree, weakest_node_info)

        return weakest_node_info

    def _prune_tree(self, tree, weakest_node):
        if self._is_leaf_node(tree):
            return tree

        decision_node = next(iter(tree))
        left_subtree, right_subtree = tree[decision_node]
        left_subtree_index, right_subtree_index = 0, 1
        _, leaf_node = weakest_node.split('as_leaf ')

        if weakest_node is decision_node:
            tree = weakest_node
        if weakest_node in left_subtree:
            tree[decision_node][left_subtree_index] = leaf_node
        if weakest_node in right_subtree:
            tree[decision_node][right_subtree_index] = leaf_node

        self._prune_tree(left_subtree, weakest_node)
        self._prune_tree(right_subtree, weakest_node)

        return tree

    def cost_complexity_pruning_path(self, X: pd.DataFrame, y: pd.Series):
        tree = self._grow_tree(X, y)   # grow a full tree
        tree_error_rate, _ = self._tree_error_rate_info(tree, [])
        error_rates = [tree_error_rate]
        ccp_alpha_list = [0.0]

        while not self._is_leaf_node(tree):
            initial_node = [None, np.inf]
            weakest_node, ccp_alpha = self._find_weakest_node(tree, initial_node)
            tree = self._prune_tree(tree, weakest_node)
            tree_error_rate, _ = self._tree_error_rate_info(tree, [])

            error_rates.append(tree_error_rate)
            ccp_alpha_list.append(ccp_alpha)

        return np.array(ccp_alpha_list), np.array(error_rates)

    def _ccp_tree_error_rate(self, tree_error_rate, num_leaf_nodes):

        return tree_error_rate + self.ccp_alpha*num_leaf_nodes   # regularization

    def _optimal_tree(self, X, y):
        tree = self._grow_tree(X, y)   # grow a full tree
        min_RT_alpha, final_tree = np.inf, None

        while not self._is_leaf_node(tree):
            RT, num_leaf_nodes = self._tree_error_rate_info(tree, [])
            current_RT_alpha = self._ccp_tree_error_rate(RT, num_leaf_nodes)

            if current_RT_alpha <= min_RT_alpha:
                min_RT_alpha = current_RT_alpha
                final_tree = deepcopy(tree)

            initial_node = [None, np.inf]
            weakest_node, _ = self._find_weakest_node(tree, initial_node)
            tree = self._prune_tree(tree, weakest_node)

        return final_tree

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.tree = self._optimal_tree(X, y)

    def _traverse_tree(self, sample, tree):
        if self._is_leaf_node(tree):
            leaf, *_ = tree.split()
            return leaf

        decision_node = next(iter(tree))  # dict key
        left_node, right_node = tree[decision_node]
        feature, other = decision_node.split(' <=')
        threshold, *_ = other.split()
        feature_value = sample[feature]

        if np.float128(feature_value) <= np.float128(threshold):
            next_node = self._traverse_tree(sample, left_node)    # left_node
        else:
            next_node = self._traverse_tree(sample, right_node)   # right_node

        return next_node

    def predict(self, samples: pd.DataFrame):
        # apply traverse_tree method for each row in a dataframe
        results = samples.apply(self._traverse_tree, args=(self.tree,), axis=1)

        return np.array(results.astype(self._y_dtype))


def tree_plot(sklearn_tree, Xa_train):
    plt.figure(figsize=(12, 18))  # customize according to the size of your tree
    plot_tree(sklearn_tree, feature_names=Xa_train.columns, filled=True, precision=6)
    plt.show()


def tree_scores_plot(estimator, ccp_alphas, train_data, test_data, metric, labels):
    train_scores, test_scores = [], []
    X_train, y_train = train_data
    X_test, y_test = test_data
    x_label, y_label = labels

    for ccp_alpha_i in ccp_alphas:
        estimator.ccp_alpha = ccp_alpha_i
        estimator.fit(X_train, y_train)
        train_pred_res = estimator.predict(X_train)
        test_pred_res = estimator.predict(X_test)

        train_score = metric(y_train, train_pred_res)
        test_score = metric(y_test, test_pred_res)
        train_scores.append(train_score)
        test_scores.append(test_score)

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label} for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()


def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, title=None):
    if y.dtype != 'int':
        y = pd.Series(LabelEncoder().fit_transform(y))
        y_train = pd.Series(LabelEncoder().fit_transform(y_train))

    feature1_name, feature2_name = X.columns[feature_indexes]
    X_feature_columns = X.values[:, feature_indexes]
    X_train_feature_columns = X_train.values[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train.values)

    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)
    
    
#https://www.kaggle.com/datasets/himanshunakrani/iris-dataset
df_path = "/Users/adury/Desktop/iris.csv"
iris = pd.read_csv(df_path)
X1, y1 = iris.iloc[:, :-1], iris.iloc[:, -1]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)
print(iris)

X2, y2 = load_linnerud(return_X_y=True, as_frame=True)
y2 = y2['Pulse']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
print(X2, y2, sep='\n')


tree_classifier = DecisionTreeCART()
tree_classifier.fit(X1_train, y1_train)
clf_ccp_alphas, _ = tree_classifier.cost_complexity_pruning_path(X1_train, y1_train)
clf_ccp_alphas = clf_ccp_alphas[:-1]

sk_tree_classifier = DecisionTreeClassifier(random_state=0)
sk_tree_classifier.fit(X1_train, y1_train)
sk_clf_path = sk_tree_classifier.cost_complexity_pruning_path(X1_train, y1_train)
sk_clf_ccp_alphas = sk_clf_path.ccp_alphas[:-1]

sk_clf_estimator = DecisionTreeClassifier(random_state=0)
train1_data, test1_data = [X1_train, y1_train], [X1_test, y1_test]
metric = accuracy_score
labels = ['Alpha', 'Accuracy']

pprint(tree_classifier.tree, width=180)
tree_plot(sk_tree_classifier, X1_train)
print(f'tree alphas: {clf_ccp_alphas}', f'sklearn alphas: {sk_clf_ccp_alphas}', sep='\n')
tree_scores_plot(sk_clf_estimator, clf_ccp_alphas, train1_data, test1_data, metric, labels)


tree_clf_prediction = tree_classifier.predict(X1_test)
tree_clf_accuracy = accuracy_score(y1_test, tree_clf_prediction)
sk_tree_clf_prediction = sk_tree_classifier.predict(X1_test)
sk_clf_accuracy = accuracy_score(y1_test, sk_tree_clf_prediction)

best_clf_ccp_alpha = 0.0143 # based on a plot
best_tree_classifier = DecisionTreeCART(ccp_alpha=best_clf_ccp_alpha)
best_tree_classifier.fit(X1_train, y1_train)
best_tree_clf_prediction = best_tree_classifier.predict(X1_test)
best_tree_clf_accuracy = accuracy_score(y1_test, best_tree_clf_prediction)

best_sk_tree_classifier = DecisionTreeClassifier(random_state=0, ccp_alpha=best_clf_ccp_alpha)
best_sk_tree_classifier.fit(X1_train, y1_train)
best_sk_tree_clf_prediction = best_sk_tree_classifier.predict(X1_test)
best_sk_clf_accuracy = accuracy_score(y1_test, best_sk_tree_clf_prediction)

print('tree prediction', tree_clf_prediction, ' ', sep='\n')
print('sklearn prediction', sk_tree_clf_prediction, ' ', sep='\n')
print('best tree prediction', best_tree_clf_prediction, ' ', sep='\n')
print('best sklearn prediction', best_sk_tree_clf_prediction, ' ', sep='\n')

pprint(best_tree_classifier.tree, width=180)
tree_plot(best_sk_tree_classifier, X1_train)
print(f'our tree pruning accuracy: before {tree_clf_accuracy} -> after {best_tree_clf_accuracy}')
print(f'sklearn tree pruning accuracy: before {sk_clf_accuracy} -> after {best_sk_clf_accuracy}')


feature_indexes = [2, 3]
title1 = 'Classification tree surface before pruning'
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_tree_classifier, feature_indexes, title1)

feature_indexes = [2, 3]
title2 = 'Classification tree surface after pruning'
decision_boundary_plot(X1, y1, X1_train, y1_train, best_sk_tree_classifier, feature_indexes, title2)

feature_indexes = [2, 3]
plt.figure(figsize=(10, 15))

for i, alpha in enumerate(clf_ccp_alphas):
    sk_tree_clf = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    plt.subplot(3, 2, i + 1)
    plt.subplots_adjust(hspace=0.5)
    title = f'ccp_alpha = {alpha}'
    decision_boundary_plot(X1, y1, X1_train, y1_train, sk_tree_clf, feature_indexes, title)