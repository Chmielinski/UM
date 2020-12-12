import pandas as pd
import sklearn.feature_selection as f_selection
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from scipy.stats import ttest_rel
import numpy as np
import sys
from sklearn.svm import SVR
from numpy import genfromtxt
import arff
from os import listdir

alpha = .05
vThreshold = 0.8

datasets_files = listdir('datasets')

feature_selection_methods = [
    { 'number': '1', 'name': 'SelectKBest' },
    { 'number': '2', 'name': 'VarianceThreshold' },
    { 'number': '3', 'name': 'RFE' },
]

classificators = [
    { 'number': '1', 'clf': KNeighborsClassifier(n_neighbors=10, metric='manhattan'), 'name': 'KNeighborsClassifier' },
    { 'number': '2', 'clf': PassiveAggressiveClassifier(random_state=50), 'name': 'PassiveAggressiveClassifier' },
    { 'number': '3', 'clf': MLPClassifier(random_state=100), 'name': 'MLPClassifier' },
    { 'number': '4', 'clf': ComplementNB(), 'name': 'ComplementNB' },
]

def read_file(filename):
    path = 'datasets/' + filename
    dataset = arff.load(open(path, 'r'))
    data = np.transpose(np.array(dataset['data']))
    features = data[:-1]
    classes = data[-1]
    return (features, classes)

def get_features(X, y):
    if feature_selection_method == '1':
        # SelectKBest
        feature_scores = f_selection.SelectKBest(chi2, k=len(X)/2).fit(X, y)
        return feature_scores
    elif feature_selection_method == '2':
        # VarianceThreshold
        feature_scores = f_selection.VarianceThreshold(threshold=vThreshold).fit_transform(X)
        return feature_scores
    elif feature_selection_method == '3':
        # RFE
        estimator = SVR(kernel="linear")
        feature_scores = f_selection.RFE(estimator, step=1).fit_transform(X, y)
        return feature_scores

def get_score(data_train, classes_train, data_test, classes_test, clf):
    clf.fit(data_train, classes_train)
    predicted = clf.predict(data_test)
    return clf.score(data_test, classes_test)


for (id, filename) in enumerate(datasets_files):
    print(filename)
    data, classes = read_file(filename)
    features = get_features(data, classes)

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=25)
    results = np.zeros((len(selected_classificators), 10))
####
    for fold, (train, test) in enumerate(rskf.split(features, classes)):
        data_train, data_test = features[train], features[test]
        classes_train, classes_test = classes[train], classes[test]

        for i in range(len(selected_classificators)):
            results[i, fold] = get_score(data_train, classes_train, data_test, classes_test, selected_classificators[i]['clf'])

    for i in range(len(selected_classificators)):
        print 'Scores for', selected_classificators[i]['name']
        print results[i]
        print

    np.savetxt('results.csv', np.asarray(results), delimiter='\t')
    test = ttest_rel(results[0], results[1])
    pval = test.pvalue
    T = test.statistic
    print 'p =', pval
    print 'T =', T
    print
    if pval > alpha:
    	print('Result undetermined')
    elif T > 0:
    	print selected_classificators[0]['name'], 'is better'
    else:
    	print selected_classificators[1]['name'], 'is better'