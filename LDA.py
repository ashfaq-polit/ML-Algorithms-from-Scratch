#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh as sp_eigh
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.plotting import plot_decision_regions


class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes, cls_counts = np.unique(y, return_counts=True)
        priors = cls_counts / n_samples

        X_cls_mean = np.array([X[y == cls].mean(axis=0) for cls in classes])
        between_cls_deviation = X_cls_mean - X.mean(axis=0)
        within_cls_deviation = X - X_cls_mean[y]

        Sb = priors * between_cls_deviation.T @ between_cls_deviation
        Sw = within_cls_deviation.T @ within_cls_deviation / n_samples
        evals, evecs = sp_eigh(Sb, Sw)
        self.dvecs = evecs[:, np.argsort(evals)[::-1]]   # discriminant vectors

        self.weights = X_cls_mean @ self.dvecs @ self.dvecs.T
        self.bias = np.log(priors) - 0.5 * np.diag(X_cls_mean @ self.weights.T)

        if self.n_components is None:
            self.n_components = min(classes.size - 1, n_features)

    def transform(self, X):
        return X @ self.dvecs[:, : self.n_components]

    def predict(self, X_test):
        scores = X_test @ self.weights.T + self.bias

        return np.argmax(scores, axis=1)
    
    
#https://www.kaggle.com/datasets/brsdincer/star-type-classification
df_path = "/Users/adury/Desktop/Stars.csv"
star_type = pd.read_csv(df_path)
print(star_type.head())

X1, y1 = star_type.iloc[:, :-1], star_type.iloc[:, -1]
cat_features_list = X1.select_dtypes(include=['object']).columns
X1[cat_features_list] = X1[cat_features_list].apply(LabelEncoder().fit_transform)

X1, y1 = X1.values, LabelEncoder().fit_transform(y1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)


lda = LDA()
lda.fit(X1_train, y1_train)
lda_pred_res = lda.predict(X1_test)
transformed = lda.transform(X1_train)
lda_accuracy = accuracy_score(y1_test, lda_pred_res)

print(f'LDA accuracy: {lda_accuracy}')
print(f'prediction: {lda_pred_res}')
print('Transformed features', transformed[:5].T, sep='\n')



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: Create a toy 3-class dataset in 3D
X, y = make_classification(
    n_samples=200,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

# Step 2: Fit LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Step 3: Plot before and after dimensionality reduction
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Original 3D data (we’ll show 2D projection for simplicity)
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=30)
ax[0].set_title("Original Data (3D projected to 2D)")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")

# LDA projection (2D)
ax[1].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='rainbow', s=30)
ax[1].set_title("After LDA Projection (2D)")
ax[1].set_xlabel("LDA Component 1")
ax[1].set_ylabel("LDA Component 2")

plt.tight_layout()
plt.show()
