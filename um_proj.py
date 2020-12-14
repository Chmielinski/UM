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
from scipy.stats import ttest_rel
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
wb = Workbook()

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
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        model = f_selection.SelectFromModel(clf, prefit=True)
        feature_scores = model.transform(X)
        return feature_scores
    else:
        raise ValueError("invalid fsm")
def get_score(data_train, classes_train, data_test, classes_test, clf):
    clf.fit(data_train, classes_train)
    predicted = clf.predict(data_test)
    return clf.score(data_test, classes_test)


for (id, filename) in enumerate(datasets_files):
    print('\nFile ' + filename +  ' (' + str(id + 1) + '/' + str(len(datasets_files)) + ')')
    sheet = wb.add_sheet(filename, cell_overwrite_ok=True)
    data, classes = read_file(filename)
    for (fsm_id, fsm) in enumerate(feature_selection_methods):
        print('\tFSM ' + fsm['name'] + ' (' + fsm['number'] + '/' + str(len(feature_selection_methods)) + ')')
        features = get_features(data, classes, fsm['number'])
        print('\tselected ' + str(len(features[0])) + ' features')
        
        fsm_sheet_position = fsm_id * len(classificators) * 3
        sheet.write(fsm_sheet_position, 0, fsm['name'])

        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=25)
        results = np.zeros((len(classificators), 10))
        
        for fold, (train, test) in enumerate(rskf.split(features, classes)):
            data_train, data_test = features[train], features[test]
            classes_train, classes_test = classes[train], classes[test]

            for (clf_id, clf) in enumerate(classificators):
                results[clf_id, fold] = get_score(data_train, classes_train, data_test, classes_test, clf['clf'])
                sheet.write(1 + fsm_sheet_position + clf_id * 2, 0, clf['name'])
                sheet.write(2 + fsm_sheet_position + clf_id * 2, fold, results[clf_id, fold])

print('\nsaving...')
date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
wb.save('results_' + date_string + '.xls')
print('Done!')
#        test = ttest_rel(results[0], results[1])
#        pval = test.pvalue
#        T = test.statistic
#        print 'p =', pval
#        print 'T =', T
#        print
#        if pval > alpha:
#            print('Result undetermined')
#        elif T > 0:
#            print selected_classificators[0]['name'], 'is better'
#        else:
#            print selected_classificators[1]['name'], 'is better'