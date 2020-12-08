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

# launch: python ./<filename> <feature_selection_method> <classificator_1> <classificator_2>

alpha = .05
kBestFeatures = 6
vThreshold = 0.8

feature_selection_method = sys.argv[1]

feature_selection_methods = [
    { 'number': '1', 'name': 'SelectKBest' },
    { 'number': '2', 'name': 'VarianceThreshold' },
    { 'number': '3', 'name': 'RFE' },
]

classificator_1 = sys.argv[2]
classificator_2 = sys.argv[3]

classificators = [
    { 'number': '1', 'clf': KNeighborsClassifier(n_neighbors=10, metric='manhattan'), 'name': 'KNeighborsClassifier' },
    { 'number': '2', 'clf': PassiveAggressiveClassifier(), 'name': 'PassiveAggressiveClassifier' },
    { 'number': '3', 'clf': MLPClassifier(), 'name': 'MLPClassifier' },
    { 'number': '4', 'clf': ComplementNB(), 'name': 'ComplementNB' },
]

fsm = next(iter(filter(lambda x: x['number'] == feature_selection_method, feature_selection_methods)), None)
if fsm is None:
    raise ValueError("Invalid argument for feature selection method")

selected_classificators = [
    next(iter(filter(lambda x: x['number'] == classificator_1, classificators)), None),
    next(iter(filter(lambda x: x['number'] == classificator_2, classificators)), None)
]

for i in range(len(selected_classificators)):
    if selected_classificators[i] is None:
        raise ValueError("Invalid argument for classificator " + str(i + 1))

print
print 'Feature selection method:', fsm['name']
print 'Classificators: ', selected_classificators[0]['name'], 'vs', selected_classificators[1]['name']
print

wines_columns = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol',
    'quality'
]

def open_file():
    temp = pd.read_csv('wines.csv', sep=';')
    temp.columns = wines_columns
    temp.drop(columns=['quality'], inplace=True)
    return temp.to_numpy()

def get_classes():
    temp = pd.read_csv('wines.csv', sep=';')
    temp.columns = wines_columns
    classes = pd.DataFrame(temp['quality']).dropna()
    return classes.to_numpy().ravel()

def get_features(X, y):
    if feature_selection_method == '1':
        # SelectKBest
        feature_scores = f_selection.SelectKBest(chi2, k=kBestFeatures).fit(X, y)
        scores_with_names = [{'name': name, 'score': round(score, 3)} for name, score in
                             zip(wines_columns[:len(wines_columns) - 1], feature_scores.scores_)]
        ranking = sorted(scores_with_names, key=lambda x: x['score'], reverse=True)
        best_features = []
        for i in range(0, kBestFeatures):
            best_features.append(ranking[i].get('name'))
        return data_ran(best_features)
    elif feature_selection_method == '2':
        # VarianceThreshold
        feature_scores = f_selection.VarianceThreshold(threshold=vThreshold).fit_transform(X)
        return feature_scores
    elif feature_selection_method == '3':
        # RFE
        estimator = SVR(kernel="linear")
        feature_scores = f_selection.RFE(estimator, step=1).fit_transform(X, y)
        return feature_scores

def data_ran(best):
    temp = pd.read_csv('wines.csv', sep=';')
    temp.columns = wines_columns
    temp1 = temp[best]
    return temp1.to_numpy()

def get_score(data_train, classes_train, data_test, classes_test, clf):
    clf.fit(data_train, classes_train)
    predicted = clf.predict(data_test)
    return clf.score(data_test, classes_test)

data = open_file()
classes = get_classes()
features = get_features(data, classes)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=25)

average_quality = 0
best_quality = 0
results = np.zeros((len(selected_classificators), 10))

for fold, (train, test) in enumerate(rskf.split(features, classes)):
    data_train, data_test = features[train], features[test]
    classes_train, classes_test = classes[train], classes[test]

    for i in range(len(selected_classificators)):
        results[i, fold] = get_score(data_train, classes_train, data_test, classes_test, selected_classificators[i]['clf'])

for i in range(len(selected_classificators)):
    print 'Scores for', selected_classificators[i]['name']
    print results[i]
    print

np.savetxt('results.csv', np.asarray(results), delimiter=',')
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