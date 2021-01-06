import pandas as pd
import sklearn.feature_selection as f_selection
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata, ranksums
import numpy as np
import sys
from sklearn.svm import SVR
from numpy import genfromtxt
import arff
from os import listdir
from xlwt import Workbook
import math
from datetime import datetime

alpha = .05
vThreshold = 0.8
n_splits = 2
n_repeats = 5

datasets_files = listdir('datasets')

feature_selection_methods = [
    { 'number': '1', 'name': 'SelectKBest' },
    { 'number': '2', 'name': 'VarianceThreshold' },
    { 'number': '3', 'name': 'SelectFromModel' },
]

classificators = [
    { 'number': '1', 'clf': KNeighborsClassifier(n_neighbors=10, metric='manhattan'), 'name': 'KNeighborsClassifier' },
    { 'number': '2', 'clf': PassiveAggressiveClassifier(random_state=50), 'name': 'PassiveAggressiveClassifier' },
    { 'number': '3', 'clf': MLPClassifier(random_state=100, max_iter=5000), 'name': 'MLPClassifier' },
    { 'number': '4', 'clf': GaussianNB(), 'name': 'GaussianNB' },
]

results = np.zeros((len(classificators), len(feature_selection_methods), len(datasets_files), n_splits * n_repeats))

def read_file(filename):
    path = 'datasets/' + filename
    dataset = arff.load(open(path, 'r'))
    data = np.transpose(np.array(dataset['data']))
    le = LabelEncoder()
    for (i, elem) in enumerate(data):
        data[i] = le.fit_transform(elem)
    data = np.array(data, np.int32)
    features = data[:-1]
    classes = data[-1]
    return (np.transpose(features), classes)

def get_features(X, y, fsm):
    if fsm == '1':
        # SelectKBest
        kBest = math.ceil(len(X[0])/2)
        feature_scores = f_selection.SelectKBest(chi2, k=kBest).fit_transform(X, y)
        return feature_scores
    elif fsm == '2':
        # VarianceThreshold
        feature_scores = f_selection.VarianceThreshold(threshold=vThreshold).fit_transform(X)
        return feature_scores
    elif fsm == '3':
        # SelectFromModel
        clf = ExtraTreesClassifier(random_state=200)
        clf = clf.fit(X, y)
        model = f_selection.SelectFromModel(clf).fit(X, y)
        feature_scores = model.transform(X)
        return feature_scores
    else:
        raise ValueError("invalid fsm")
def get_score(data_train, classes_train, data_test, classes_test, clf):
    clf.fit(data_train, classes_train)
    predicted = clf.predict(data_test)
    return clf.score(data_test, classes_test)

wb = Workbook()

for (dataset_id, filename) in enumerate(datasets_files):
    print('\nFile ' + filename +  ' (' + str(dataset_id + 1) + '/' + str(len(datasets_files)) + ')')
    sheet = wb.add_sheet(filename, cell_overwrite_ok=True)
    data, classes = read_file(filename)
    for (fsm_id, fsm) in enumerate(feature_selection_methods):
        print('\tFSM ' + fsm['name'] + ' (' + fsm['number'] + '/' + str(len(feature_selection_methods)) + ')')
        features = get_features(data, classes, fsm['number'])
        print('\tselected ' + str(len(features[0])) + ' features')
        
        fsm_sheet_position = fsm_id * len(classificators) * 3
        sheet.write(fsm_sheet_position, 0, fsm['name'])

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=25)
        
        for fold, (train, test) in enumerate(rskf.split(features, classes)):
            data_train, data_test = features[train], features[test]
            classes_train, classes_test = classes[train], classes[test]

            for (clf_id, clf) in enumerate(classificators):
                results[clf_id, fsm_id, dataset_id, fold] = get_score(data_train, classes_train, data_test, classes_test, clf['clf'])
                sheet.write(1 + fsm_sheet_position + clf_id * 2, 0, clf['name'])
                sheet.write(2 + fsm_sheet_position + clf_id * 2, fold, results[clf_id, fsm_id, dataset_id, fold])

print('\nsaving results...')
date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
wb.save('results_' + date_string + '.xls')
print('Done!')

# [fsm, clf, dataset, fold]

wb = Workbook()

for (clf_id, values) in enumerate(results):
    p_value_sheet_offset = 8
    sheet = wb.add_sheet(classificators[clf_id]['name'])
    sheet.write(0, 0, 'w-statistic')
    sheet.write(p_value_sheet_offset, 0, 'p-value')
    sheet.write(2 * p_value_sheet_offset, 0, 'Advantage')
    sheet.write(3 * p_value_sheet_offset, 0, 'Statistical significance')
    for (fsm_id, fsm) in enumerate(feature_selection_methods):
        for i in range(4):
            sheet.write(i * p_value_sheet_offset, fsm_id + 1, fsm['name'])
            sheet.write(i * p_value_sheet_offset + fsm_id + 1, 0, fsm['name'])
    
    mean_scores = np.mean(values, axis=2).T
    print(mean_scores)
    ranks = []

    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print(ranks)

    w_stat = np.zeros((len(feature_selection_methods), len(feature_selection_methods)))
    p_val = np.zeros((len(feature_selection_methods), len(feature_selection_methods)))
    advantage = np.zeros((len(feature_selection_methods), len(feature_selection_methods)))
    significance = np.zeros((len(feature_selection_methods), len(feature_selection_methods)))

    for i in range(len(feature_selection_methods)):
        for j in range(len(feature_selection_methods)):
            w_stat[i, j], p_val[i, j] = ranksums(ranks.T[i], ranks.T[j])

    advantage[w_stat > 0] = 1
    significance[p_val <= alpha] = 1
    
    for i in range(len(feature_selection_methods)):
        for j in range(len(feature_selection_methods)):
            sheet.write(i + 1, j + 1, w_stat[i, j])
            sheet.write(p_value_sheet_offset + i + 1, j + 1, p_val[i, j])
            sheet.write(2 * p_value_sheet_offset + i + 1, j + 1, advantage[i, j])
            sheet.write(3 * p_value_sheet_offset + i + 1, j + 1, significance[i, j])

print('\nsaving test results...')
date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
wb.save('test_' + date_string + '.xls')
print('Done!')