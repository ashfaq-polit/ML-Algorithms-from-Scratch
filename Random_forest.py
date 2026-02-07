#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, VotingClassifier, ExtraTreesClassifier
from joblib import Parallel, delayed
from mlxtend.plotting import plot_decision_regions


class RandomForest:
    def __init__(self, regression=False, n_estimators=100, max_depth=None,
                 max_features=1.0, n_jobs=-1, random_state=0, ccp_alpha=0.0):
        self.regression = regression
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.trained_trees_info = []
        self.general_random = np.random.RandomState(self.random_state)

    #bootstrapping with random subspaces method
    def _rsm_bootstrapping(self, X, y):
        n_samples, n_features = X.shape
        if self.regression:
            max_features = self.max_features * n_features
        else:
            max_features = np.sqrt(n_features)

        sample_indexes = self.general_random.choice(n_samples, n_samples)
        features = self.general_random.choice(X.columns, round(max_features))
        X_b, y_b = X.iloc[sample_indexes][features], y.iloc[sample_indexes]

        return X_b, y_b

    def _train_tree(self, X, y):
        if self.regression:
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state,
                                         ccp_alpha=self.ccp_alpha)
        else:
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self.random_state,
                                          ccp_alpha=self.ccp_alpha)

        return tree.fit(X, y), X.columns

    def fit(self, X, y):
        boot_data = (self._rsm_bootstrapping(X, y) for _ in range(self.n_estimators))
        train_trees = (delayed(self._train_tree)(X_b, y_b) for X_b, y_b in boot_data)
        self.trained_trees_info = Parallel(n_jobs=self.n_jobs)(train_trees)

    def predict(self, samples):
        prediction = (delayed(tree_i.predict)(samples[tree_i_features])
                      for (tree_i, tree_i_features) in self.trained_trees_info)

        trees_predictions = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(prediction))

        if self.regression:
            forest_prediction = trees_predictions.mean(axis=0)
        else:
            forest_prediction = trees_predictions.mode(axis=0).iloc[0]

        return np.array(forest_prediction)
    
    
def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, title=None):
    feature1_name, feature2_name = X.columns[feature_indexes]
    X_feature_columns = X.values[:, feature_indexes]
    X_train_feature_columns = X_train.values[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train.values)

    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)


#https://www.kaggle.com/datasets/uciml/glass
df_path = "/Users/adury/Desktop/glass.csv"
glass_df = pd.read_csv(df_path)
X1, y1 = glass_df.iloc[:, :-1], glass_df.iloc[:, -1]
y1 = pd.Series(LabelEncoder().fit_transform(y1))
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)
print(glass_df)

X2, y2 = load_diabetes(return_X_y=True, as_frame=True)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=0)
print(X2, y2, sep='\n')


random_forest_classifier = RandomForest(random_state=0)
random_forest_classifier.fit(X1_train, y1_train)
rf_clf_pred_res = random_forest_classifier.predict(X1_test)
rf_clf_accuracy = accuracy_score(y1_test, rf_clf_pred_res)
print(rf_clf_pred_res)
print(f'rf_clf accuracy: {rf_clf_accuracy}')

sk_random_forest_classifier = RandomForestClassifier(random_state=0)
sk_random_forest_classifier.fit(X1_train, y1_train)
sk_rf_clf_pred_res = sk_random_forest_classifier.predict(X1_test)
sk_rf_clf_accuracy = accuracy_score(y1_test, sk_rf_clf_pred_res)
print(sk_rf_clf_pred_res)
print(f'sk_rf_clf accuracy: {sk_rf_clf_accuracy}')


random_forest_regressor = RandomForest(regression=True, random_state=0)
random_forest_regressor.fit(X2_train, y2_train)
rf_reg_pred_res = random_forest_regressor.predict(X2_test)
mape = mean_absolute_percentage_error(y2_test, rf_reg_pred_res)
print(f'mape: {mape}')
print(rf_reg_pred_res)

sk_random_forest_regressor = RandomForestRegressor(random_state=0)
sk_random_forest_regressor.fit(X2_train, y2_train)
sk_rf_reg_pred_res = sk_random_forest_regressor.predict(X2_test)
sk_mape = mean_absolute_percentage_error(y2_test, sk_rf_reg_pred_res)
print(f'sk_mape: {sk_mape}')
print(sk_rf_reg_pred_res)


importances = sk_random_forest_regressor.feature_importances_
feature_importances = pd.Series(importances, index=X2_train.columns)
feature_importances.sort_values(ascending=True).plot(kind='barh')


sk_tree_clf = DecisionTreeClassifier(random_state=0)
sk_tree_clf.fit(X1_train, y1_train)
tree_pred_res = sk_tree_clf.predict(X1_test)
tree_accuracy = accuracy_score(y1_test, tree_pred_res)
print(f'tree accuracy: {tree_accuracy}')
print(tree_pred_res)

feature_indexes = [1, 3]
title1 = 'DecisionTreeClassifier surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_tree_clf, feature_indexes, title1)


estimators = [('lr', LogisticRegression(random_state=0, max_iter=10000)),
              ('dt', DecisionTreeClassifier(random_state=0)),
              ('svc', SVC(probability=True, random_state=0))]

voting_clf = VotingClassifier(estimators, voting='soft')
voting_clf.fit(X1_train, y1_train)
voting_clf_pred_res = voting_clf.predict(X1_test)
voting_clf_accuracy = accuracy_score(y1_test, voting_clf_pred_res)
print(f'voting_clf accuracy: {voting_clf_accuracy}')
print(voting_clf_pred_res)

feature_indexes = [1, 3]
title2 = 'VotingClassifier surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, voting_clf, feature_indexes, title2)


dt_clf = DecisionTreeClassifier(random_state=0)
bagging_clf = BaggingClassifier(estimator=dt_clf, n_estimators=100, random_state=0)
bagging_clf.fit(X1_train, y1_train)
bagging_pred_res = bagging_clf.predict(X1_test)
bagging_clf_accuracy = accuracy_score(y1_test, bagging_pred_res)
print(f'bagging_clf accuracy: {bagging_clf_accuracy}')
print(bagging_pred_res)

feature_indexes = [1, 3]
title3 = 'BaggingClassifier surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, bagging_clf, feature_indexes, title3)