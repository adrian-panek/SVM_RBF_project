import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SVM_copy import SVM
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

classifier = 'svm_our'
real_dataset = True

if real_dataset:
    dataset = pd.read_csv('../dataset/Cancer_Data.csv', sep=',')
    X = np.zeros((len(dataset['radius_mean'].values), 4))
    X[:,0] = dataset['radius_mean'].values
    X[:,1] = dataset['texture_mean'].values
    X[:,2] = dataset['area_mean'].values/100
    X[:,3] = dataset['smoothness_mean'].values*10
    y = np.where(dataset['diagnosis'].values == 'M', -1.0, 1.0)
else:
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, random_state=59)
    y = np.where(y == 1, 1.0, -1.0)


if classifier == 'neural_netowork':
    classifier = MLPClassifier()
    
if classifier == 'knn':
    classifier = KNeighborsClassifier()

if classifier == 'svm_sklearn':
    classifier = SVC()

if classifier == 'decision_trees':
    classifier = DecisionTreeClassifier()

if classifier == 'svm_our':
    la = 1
    classifier = SVM(la=la)

fig = plt.figure(figsize = (10,10))
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
plt.show()

classifier.fit(X, y)
pred = classifier.predict(X)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
total_acc_score = []
total_f1_score = []
total_prec_score = []

for (train_index, test_index) in rskf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier.fit(X_train, y_train)
    support_matrix = classifier.predict(X_test)
    acc_score = accuracy_score(y_test, support_matrix)
    total_acc_score.append(acc_score)
    f1_scor = f1_score(y_test, support_matrix)
    total_f1_score.append(f1_scor)
    prec_score = precision_score(y_test, support_matrix)
    total_prec_score.append(prec_score)

with open(f'../results/{classifier}_results.npy', 'wb') as f:
    np.save(f, round(np.mean(total_acc_score),3))
    np.save(f, round(np.mean(total_f1_score),3))
    np.save(f, round(np.mean(total_prec_score),3))