import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SVM import SVM, RBF
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

classifier = 'knn'
real_dataset = True

if real_dataset:
    dataset = pd.read_csv('../dataset/Cancer_Data.csv', sep=',')
    X = np.zeros((len(dataset['radius_mean'].values), 2))
    X[:,0] = dataset['radius_mean'].values
    X[:,1] = dataset['texture_mean'].values
    y = np.where(dataset['diagnosis'].values == 'M', -1, 1)
else:
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)

if classifier == 'neural_netowork':
    classifier = MLPClassifier()
    
if classifier == 'knn':
    classifier = KNeighborsClassifier()

if classifier == 'svm_sklearn':
    classifier = SVC()

if classifier == 'decision_trees':
    classifier = DecisionTreeClassifier()

if classifier == 'svm_our':
    lr = 0.1
    la = 1
    X = RBF(X, 0.5)
    classifier = SVM(lr, la, 100)

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
    x_train_2, x_test_2 = X[train_index], X[test_index]
    y_train_2, y_test_2 = y[train_index], y[test_index]
    classifier.fit(x_train_2, y_train_2)
    support_matrix = classifier.predict(x_test_2)
    acc_score = accuracy_score(y_test_2, support_matrix)
    total_acc_score.append(acc_score)
    f1_scor = f1_score(y_test_2, support_matrix)
    total_f1_score.append(f1_scor)
    prec_score = precision_score(y_test_2, support_matrix)
    total_prec_score.append(prec_score)


with open(f'../results/{classifier}_results.npy', 'wb') as f:
    np.save(f, round(np.mean(total_acc_score),3))
    np.save(f, round(np.mean(total_f1_score),3))
    np.save(f, round(np.mean(total_prec_score),3))