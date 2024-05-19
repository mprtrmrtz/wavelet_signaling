import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

import pywt
import scipy.stats

from collections import defaultdict, Counter

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
#     n5 = np.nanpercentile(list_values, 5)
#     n25 = np.nanpercentile(list_values, 25)
#     n75 = np.nanpercentile(list_values, 75)
#     n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
#     return [n5, n25, n75, n95, median, mean, std, var, rms]
    return [median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

def get_train_test(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test



ecg_signals = new_file.iloc[:,:-1].values
ecg_labels = new_file.iloc[:,-1].values

dict_ecg_data = defaultdict(list)
for ii, label in enumerate(ecg_labels):
    dict_ecg_data[label].append(ecg_signals[ii])


list_labels = []
list_features = []
for k, v in dict_ecg_data.items():
    yval = list(dict_ecg_data.keys()).index(k)
    for signal in v:
        features = []
        list_labels.append(yval)
        list_coeff = pywt.wavedec(signal, 'sym5')
        for coeff in list_coeff:
            features += get_features(coeff)
        list_features.append(features)
df = pd.DataFrame(list_features)
ycol = 'y'
xcols = list(range(df.shape[1]))
df.loc[:,ycol] = list_labels

df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, ycol, xcols, ratio = 0.7)

# K Nearest Neighbors
clf = KNeighborsClassifier(5, p=1)
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)

# Support Vector - Linear
clf = SVC(kernel="linear")
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)

# Support Vector - gamma=3, C=1
clf = SVC(gamma=3, C=1)
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)

# Decition Tree
clf = DecisionTreeClassifier(max_depth=20)
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)

# Random Forest
clf = RandomForestClassifier(max_depth=20, n_estimators=10, max_features=5)
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)

# Multi Layer Perceptron
clf = MLPClassifier(hidden_layer_sizes=(50,100), alpha=0.01, max_iter=1000, activation='tanh', solver='adam')
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)

# Ada Boost
clf = AdaBoostClassifier()
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)

# Gradient Boosting
clf = GradientBoostingClassifier(n_estimators=10000)
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("The Train Score is {}".format(train_score))
print("The Test Score is {}".format(test_score))

# Gaussian Naive Bayes
clf = GaussianNB()
print(clf)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print("training score: ", train_score)
print("testing score: ", test_score)