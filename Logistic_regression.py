#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


class GDLogisticRegression:
    def __init__(self, learning_rate=0.1, tolerance=0.0001, max_iter=1000):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias, self.weights = 0, np.zeros(n_features)
        previous_db, previous_dw = 0, np.zeros(n_features)

        for _ in range(self.max_iter):
            y_pred_linear = X @ self.weights + self.bias
            y_pred_sigmoid = 1 / (1 + np.exp(-y_pred_linear))
            db = 1 / n_samples * np.sum(y_pred_sigmoid - y)
            dw = 1 / n_samples * X.T @ (y_pred_sigmoid - y)

            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw
            abs_db_reduction = np.abs(db - previous_db)
            abs_dw_reduction = np.abs(dw - previous_dw)

            if abs_db_reduction < self.tolerance:
                if abs_dw_reduction.all() < self.tolerance:
                    break

            previous_db = db
            previous_dw = dw

    def predict(self, X_test):
        y_pred_linear = X_test @ self.weights + self.bias
        y_pred_sigmoid = 1 / (1 + np.exp(-y_pred_linear))
        classes = np.array([0 if pred < 0.5 else 1 for pred in y_pred_sigmoid])

        return classes
    
    
def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, title=None):
    feature1_name, feature2_name = X.columns[feature_indexes]
    X_feature_columns = X.values[:, feature_indexes]
    X_train_feature_columns = X_train.values[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train.values)

    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)
    
    
# https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset/versions/1
df_path = "/Users/adury/Desktop/heart_attack_prediction_dataset"
heart_attack = pd.read_csv(df_path)
print(heart_attack.head())

X1, y1 = heart_attack.iloc[:, :-1], heart_attack.iloc[:, -1]
X1_scaled = StandardScaler().fit_transform(X1)
y1 = pd.Series(LabelEncoder().fit_transform(y1))

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)
X1_train_s, X1_test_s, y1_train, y1_test = train_test_split(X1_scaled, y1, random_state=0)


logistic_regression = GDLogisticRegression()
logistic_regression.fit(X1_train_s, y1_train)
pred_res = logistic_regression.predict(X1_test_s)
accuracy = accuracy_score(y1_test, pred_res)

print(f'Logistic regression accuracy: {accuracy}')
print(f'prediction: {pred_res}')


sk_logistic_regression = LogisticRegression(penalty=None, max_iter=1000, multi_class='ovr')
sk_logistic_regression.fit(X1_train, y1_train)
sk_pred_res = sk_logistic_regression.predict(X1_test)
sk_accuracy = accuracy_score(y1_test, sk_pred_res)

print(f'sk Logistic regression accuracy: {sk_accuracy}')
print(f'prediction: {sk_pred_res}')

feature_indexes = [3, 9]
title1 = 'LogisticRegression surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_logistic_regression, feature_indexes, title1)



class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, tolerance=0.0001, max_iter=1000):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def _softmax(self, predictions):
        exp = np.exp(predictions)

        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        n_samples, n_features = X.shape
        one_hot_y = pd.get_dummies(y).to_numpy()

        self.bias = np.zeros(n_classes)
        self.weights = np.zeros((n_features, n_classes))
        previous_db = np.zeros(n_classes)
        previous_dw = np.zeros((n_features, n_classes))

        for _ in range(self.max_iter):
            y_pred_linear = X @ self.weights + self.bias
            y_pred_softmax = self._softmax(y_pred_linear)
            db = 1 / n_samples * np.sum(y_pred_softmax - one_hot_y, axis=0)   # sum by columns
            dw = 1 / n_samples * X.T @ (y_pred_softmax - one_hot_y)

            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw
            abs_db_reduction = np.abs(db - previous_db)
            abs_dw_reduction = np.abs(dw - previous_dw)

            if abs_db_reduction.all() < self.tolerance:
                if abs_dw_reduction.all() < self.tolerance:
                    break

            previous_db = db
            previous_dw = dw

    def predict(self, X_test):
        y_pred_linear = X_test @ self.weights + self.bias
        y_pred_softmax = self._softmax(y_pred_linear)
        most_prob_classes = np.argmax(y_pred_softmax, axis=1)

        return most_prob_classes

#https://www.kaggle.com/datasets/sujithmandala/credit-score-classification-dataset/data
df_path = "/Users/adury/Desktop/Credit Score Classification Dataset.csv"
credit_score = pd.read_csv(df_path)
print(credit_score.head())

X2, y2 = credit_score.iloc[:, :-1], credit_score.iloc[:, -1]
cat_features_list = X2.select_dtypes(include=['object']).columns
X2[cat_features_list] = X2[cat_features_list].apply(LabelEncoder().fit_transform)
X2_scaled = StandardScaler().fit_transform(X2)
y2 = pd.Series(LabelEncoder().fit_transform(y2))

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=0)
X2_train_s, X2_test_s, y2_train, y2_test = train_test_split(X2_scaled, y2, random_state=0)


softmax_regression = SoftmaxRegression()
softmax_regression.fit(X2_train_s, y2_train)
softmax_pred_res = softmax_regression.predict(X2_test_s)
softmax_accuracy = accuracy_score(y2_test, softmax_pred_res)

print(f'Softmax-regression accuracy: {softmax_accuracy}')
print(f'Softmax prediction: {softmax_pred_res}')

sk_softmax_regression = LogisticRegression(penalty=None, max_iter=1000, multi_class='multinomial')
sk_softmax_regression.fit(X2_train, y2_train)
sk_softmax_pred_res = sk_softmax_regression.predict(X2_test)
sk_softmax_accuracy = accuracy_score(y2_test, sk_softmax_pred_res)

print(f'sk Softmax-regression accuracy: {sk_softmax_accuracy}')
print(f'sk Softmax prediction: {sk_softmax_pred_res}')



import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-6, 6, 200)
p = 1 / (1 + np.exp(-X))  # logistic curve

plt.figure(figsize=(6,4))
plt.plot(X, p, label='Predicted Probability p')
plt.fill_between(X, p*(1-p), alpha=0.2, label='Variance = p(1-p)')
plt.title("Logistic Regression: Bernoulli Model for Binary y")
plt.xlabel("Linear predictor (Xβ)")
plt.ylabel("Probability of y=1")
plt.legend()
plt.grid(True)
plt.show()
