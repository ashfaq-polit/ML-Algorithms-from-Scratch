#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from mlxtend.plotting import plot_decision_regions


class AdaBoost:
    def __init__(self, regression=False, n_estimators=50, learning_rate=1.0, random_state=0):
        self.regression = regression
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.stumps = []
        self.weights = []

    def _update_weights(self, args):
        if self.regression:
            weights, beta, loss_function = args
            weights *= beta ** (1 - loss_function)
        else:
            weights, wrong_predictions, stump_weight = args
            weights[wrong_predictions] *= np.exp(stump_weight)

        return weights

    @staticmethod
    def _normalize(weights: np.ndarray):
        return weights / sum(weights)

    def fit(self, X, y):
        n_samples = len(y)
        self.K = len(np.unique(y.values))   # num unique classes
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            if self.regression:
                stump = DecisionTreeRegressor(max_depth=3, random_state=self.random_state)
                stump_predictions = stump.fit(X, y, sample_weight=weights).predict(X)

                abs_instance_errors = abs(stump_predictions - y)
                adjusted_instance_errors = abs_instance_errors / max(abs_instance_errors)
                adjusted_stump_error = sum(weights * adjusted_instance_errors)

                if adjusted_stump_error >= 0.5:
                    self.stumps.pop(-1)
                    break

                beta = adjusted_stump_error / (1 - adjusted_stump_error)
                stump_weight = self.learning_rate * np.log(1 / beta)
                args = [weights, beta, adjusted_instance_errors]
            else:
                stump = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
                stump_predictions = stump.fit(X, y, sample_weight=weights).predict(X)

                wrong_predictions = stump_predictions != y
                r = sum(weights[wrong_predictions])   # weighted_error_rate
                stump_weight = self.learning_rate * np.log((1 - r) / r) + np.log(self.K - 1)
                args = [weights, wrong_predictions, stump_weight]

            self.stumps.append(stump)
            self.weights.append(stump_weight)
            weights= self._update_weights(args)
            weights = self._normalize(weights)

    def _max_weighted_votes(self, samples):
        n_samples = len(samples)
        sample_indexes = np.array(range(n_samples))
        prediction_weights = np.zeros((n_samples, self.K))

        for i in range(self.n_estimators):
            stump_prediction = self.stumps[i].predict(samples)
            prediction_weights[sample_indexes, stump_prediction] += self.weights[i]

        return np.argmax(prediction_weights, axis=1)

    def _weighted_median_prediction(self, samples):
        n_samples = len(samples)
        sample_indexes = np.arange(n_samples)
        predictions = np.array([stump.predict(samples) for stump in self.stumps]).T
        sorted_pred_indexes = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        cumsum_weights = np.array(self.weights)[sorted_pred_indexes].cumsum(axis=1)
        is_over_median = cumsum_weights >= 0.5 * sum(self.weights)   # True/False matrix
        median_indexes = is_over_median.argmax(axis=1)
        median_prediction_indexes = sorted_pred_indexes[sample_indexes, median_indexes]

        return predictions[sample_indexes, median_prediction_indexes]

    def predict(self, samples):
        if self.regression:
            return self._weighted_median_prediction(samples)
        else:
            return self._max_weighted_votes(samples)
        
        
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


adaboost_clf = AdaBoost(random_state=0)
adaboost_clf.fit(X1_train, y1_train)
adaboost_clf_pred_res = adaboost_clf.predict(X1_test)
adaboost_clf_accuracy = accuracy_score(y1_test, adaboost_clf_pred_res)
print(f'adaboost_clf_accuracy: {adaboost_clf_accuracy}')
print(adaboost_clf_pred_res, '', sep='\n')

sk_adaboost_clf = AdaBoostClassifier(random_state=0, algorithm='SAMME')
sk_adaboost_clf.fit(X1_train, y1_train)
sk_adaboost_clf_pred_res = sk_adaboost_clf.predict(X1_test)
sk_adaboost_clf_accuracy = accuracy_score(y1_test, sk_adaboost_clf_pred_res)
print(f'sk_adaboost_clf_accuracy: {sk_adaboost_clf_accuracy}')
print(sk_adaboost_clf_pred_res)

feature_indexes = [1, 3]
title1 = 'AdaBoostClassifier surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_adaboost_clf, feature_indexes, title1)


adaboost_reg = AdaBoost(regression=True, random_state=0)
adaboost_reg.fit(X2_train, y2_train)
adaboost_reg_pred_res = adaboost_reg.predict(X2_test)
mape = mean_absolute_percentage_error(y2_test, adaboost_reg_pred_res)
print(f'adaboost_mape {mape}')
print(adaboost_reg_pred_res, '', sep='\n')

sk_adaboost_reg = AdaBoostRegressor(random_state=0, loss='linear')
sk_adaboost_reg.fit(X2_train, y2_train)
sk_adaboost_reg_pred_res = sk_adaboost_reg.predict(X2_test)
sk_mape = mean_absolute_percentage_error(y2_test, sk_adaboost_reg_pred_res)
print(f'sk_adaboost_mape {sk_mape}')
print(sk_adaboost_reg_pred_res)