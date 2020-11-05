# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:07:38 2020

@author: tchat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import ensemble
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
X_train = np.load("data/ml/train_swir_nr.npy")
X_test = np.load("data/ml/test_swir_nr.npy")
Y_train = np.load("data/ml/train_concentration.npy")

# Do train validation data split in 80-20 ratio
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)

# Standardization of data
sc = StandardScaler()
X_Train_std = sc.fit_transform(X_Train)
X_Val_std = sc.transform(X_Val)

# List to store various accuracy scores
RScore = []

# The Algorithms
classifiers = [
        svm.SVR(),
        linear_model.LinearRegression(),
        kernel_ridge.KernelRidge(),
        ensemble.RandomForestRegressor(),
        ensemble.GradientBoostingRegressor(),
        ensemble.AdaBoostRegressor(),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
        linear_model.PassiveAggressiveRegressor()
        ]

# Apply the algorithms
for item in classifiers:
    model = item
    model.fit(X_Train_std, Y_Train)
    Y_Val_pred = model.predict(X_Val_std)
    RScore.append(r2_score(Y_Val_pred,Y_Val))

# Find the index with the max R score    
index = np.argmax(RScore)

# Apply algorithm corresponding to index to get final predictions
clf = classifiers[index]
clf.fit(X_Train_std, Y_Train)
Y_Val_pred = clf.predict(X_Val_std)

# Plot
_, ax = plt.subplots()
ax.scatter(x = range(0, Y_Val.size), y=Y_Val, c = 'blue', label = 'Actual', alpha = 0.3)
ax.scatter(x = range(0, Y_Val_pred.size), y=Y_Val_pred, c = 'red', label = 'Predicted', alpha = 0.3)

plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('mpg')
plt.legend()
plt.savefig('Answers/Actual vs Predicted Values Plot.png')
plt.show()

# Predict y_test from test data
X_Test = np.transpose(X_test)
X_Test_std = sc.transform(X_Test)
Y_Test = clf.predict(X_Test_std)
np.save('Answers/Y_Test.npy', Y_Test)

# Apply PCA for feature selection
pca = PCA(whiten = True)
pca.fit(X_Train_std)
variance = pd.DataFrame(pca.explained_variance_ratio_)
Value = np.cumsum(pca.explained_variance_ratio_)

pca = PCA(n_components = 20, whiten = True)
pca = pca.fit(X_Train_std)
Xpca_train = pca.transform(X_Train_std)
Xpca_test = pca.transform(X_Test_std)

clf.fit(Xpca_train, Y_Train)
Ypca_test = clf.predict(Xpca_test)
