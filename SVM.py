#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state, check_array
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR, _libsvm
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes


class LinearSVM:
    def __init__(self, regression=False, C=1.0, eps=0, learning_rate=0.001, max_iter=1000,
                 random_state=0):
        self.regression = regression
        self.C = C
        self.eps = eps
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        if self.regression:
            self.bias, self.weights = self._find_weights(X, y)
        else:
            classes = np.unique(y)
            n_classes = len(classes)
            _, n_features = X.shape

            self.bias = np.zeros(n_classes)
            self.weights = np.zeros((n_classes, n_features))
            np.random.seed(self.random_state)

            for i, cls in enumerate(classes):
                y_binary = np.where(y == cls, 1, -1)
                self.bias[i], self.weights[i] = self._find_weights(X, y_binary)

    def _find_weights(self, X, y):
        n_samples, n_features = X.shape
        bias = 0
        weights = np.zeros(n_features) if self.regression else np.random.randn(n_features)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                y_pred = X[i] @ weights + bias
                margin = y[i] - y_pred if self.regression else y[i] * y_pred
                condition = np.abs(margin) > self.eps if self.regression else margin < 1

                if condition:
                    if self.regression:
                        db = -self.C * (margin - self.eps)
                        dw = -self.C * (margin - self.eps) * X[i]
                    else:
                        db = -self.C * y[i]
                        dw = -self.C * y[i] * X[i]

                    bias -= self.learning_rate * db
                    weights -= self.learning_rate * dw

        return bias, weights

    def predict(self, X):
        scores = X @ self.weights.T + self.bias

        return scores if self.regression else np.argmax(scores, axis=1)
    
    
#Kernel SVM based on Libsvm   
class SVM:
    def __init__(self, regression=False, C=1.0, kernel='rbf', degree=3, solver='auto',
                 gamma='scale', epsilon=0.1, coef0=0.0, shrinking=True, probability=False,
                 tol=0.001, cache_size=200, max_iter=-1, random_state=None):
        self.regression = regression
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.solver = solver
        self.gamma = gamma
        self.epsilon = epsilon
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        rnd = check_random_state(self.random_state)
        seed = rnd.randint(np.iinfo('i').max)

        if self.gamma == 'scale':
            self.gamma = 1.0 / (X.shape[1] * X.var()) if X.var() != 0 else 1.0
        elif self.gamma == 'auto':
                self.gamma = 1.0 / X.shape[1]
        else:
            self.gamma = self.gamma

        if self.solver == 'auto':
            self.solver = 'epsilon_svr' if self.regression else 'c_svc'

        libsvm_impl = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']
        self.solver = libsvm_impl.index(self.solver)

        (self.support_, self.support_vectors_, self._n_support, self.dual_coef_, self.intercept_,
         self._probA, self._probB, self.fit_status_, self._num_iter

        ) = _libsvm.fit(X, y, C=self.C, svm_type=self.solver, kernel=self.kernel, gamma=self.gamma,
                        degree=self.degree, epsilon=self.epsilon, coef0=self.coef0, tol=self.tol,
                        shrinking=self.shrinking, probability=self.probability,
                        cache_size=self.cache_size, max_iter=self.max_iter, random_seed=seed
                        )

    def predict(self, X_test):
        X_test = X_test.astype(np.float64)
        prediction = _libsvm.predict(X_test, self.support_, self.support_vectors_, self._n_support,
                                     self.dual_coef_, self.intercept_, self._probA, self._probB,
                                     svm_type=self.solver, kernel=self.kernel, degree=self.degree,
                                     coef0=self.coef0, gamma=self.gamma, cache_size=self.cache_size
                                     )

        return prediction if self.regression else prediction.astype(int)
    
    
def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, title=None):
    feature1_name, feature2_name = X.columns[feature_indexes]
    X_feature_columns = X.values[:, feature_indexes]
    X_train_feature_columns = X_train[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train)

    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)
    
    
#https://www.kaggle.com/datasets/himanshunakrani/iris-dataset
df_path = "/Users/adury/Desktop/iris.csv"
iris = pd.read_csv(df_path)
X1, y1 = iris.iloc[:, :-1], iris.iloc[:, -1]
y1 = pd.Series(LabelEncoder().fit_transform(y1))
X1_train, X1_test, y1_train, y1_test = train_test_split(X1.values, y1.values, test_size=0.3, random_state=0)
print(iris)

X2, y2 = load_diabetes(return_X_y=True, as_frame=True)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2.values, y2.values, random_state=0)
print(X2, y2, sep='\n')


linear_svc = LinearSVM(random_state=0)
linear_svc.fit(X1_train, y1_train)
linear_svc_pred_res = linear_svc.predict(X1_test)
linear_svc_accuracy = accuracy_score(y1_test, linear_svc_pred_res)

print(f'LinearSVC accuracy: {linear_svc_accuracy:}')
print(linear_svc_pred_res)


#LinearSVC (scikit-learn)
sk_linear_svc = LinearSVC(loss='squared_hinge', max_iter=10000, random_state=0)
sk_linear_svc.fit(X1_train, y1_train)
sk_linear_svc_pred_res = sk_linear_svc.predict(X1_test)
sk_linear_svc_accuracy = accuracy_score(y1_test, sk_linear_svc_pred_res)

print(f'sk LinearSVC accuracy: {sk_linear_svc_accuracy:}')
print(sk_linear_svc_pred_res)

feature_indexes = [2, 3]
title1 = 'LinearSVC surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_linear_svc, feature_indexes, title1)


svc = SVM(random_state=0, gamma='auto')
svc.fit(X1_train, y1_train)
svc_pred_res = svc.predict(X1_test)
svc_accuracy = accuracy_score(y1_test, svc_pred_res)

print(f'SVC accuracy: {svc_accuracy:}')
print(svc_pred_res)


sk_svc = SVC(random_state=0, gamma='auto')
sk_svc.fit(X1_train, y1_train)
sk_svc_pred_res = sk_svc.predict(X1_test)
sk_svc_accuracy = accuracy_score(y1_test, sk_svc_pred_res)

print(f'sk SVC accuracy: {sk_svc_accuracy:}')
print(sk_svc_pred_res)

feature_indexes = [2, 3]
title2 = 'SVC surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_svc, feature_indexes, title2)


linear_svr = LinearSVM(regression=True)
linear_svr.fit(X2_train, y2_train)
linear_svr_pred_res = linear_svr.predict(X2_test)
linear_svr_r2 = r2_score(y2_test, linear_svr_pred_res)

print(f'LinearSVR r2 score: {linear_svr_r2:}')
print(linear_svr_pred_res)


sk_linear_svr = LinearSVR(loss='squared_epsilon_insensitive')
sk_linear_svr.fit(X2_train, y2_train)
sk_linear_svr_pred_res = sk_linear_svr.predict(X2_test)
sk_linear_svr_r2 = r2_score(y2_test, sk_linear_svr_pred_res)

print(f'sk LinearSVR R2 score: {sk_linear_svr_r2:}')
print(sk_linear_svr_pred_res)


svr = SVM(regression=True)
svr.fit(X2_train, y2_train)
svr_pred_res = svr.predict(X2_test)
svc_r2 = r2_score(y2_test, svr_pred_res)

print(f'SVR r2 score: {svc_r2:}')
print(svr_pred_res)


sk_svr = SVR()
sk_svr.fit(X2_train, y2_train)
sk_svr_pred_res = sk_svr.predict(X2_test)
sk_svr_r2 = r2_score(y2_test, sk_svr_pred_res)

print(f'sk SVR r2 score: {sk_svr_r2:}')
print(sk_svr_pred_res)