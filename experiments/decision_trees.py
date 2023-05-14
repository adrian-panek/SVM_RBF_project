import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
trees_pred = clf.predict(X_test)

fig = plt.figure(figsize = (10,10))
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
plt.show()

f1 = f"F1 score score for Decision trees Classifier is: {(round(f1_score(y_test, trees_pred, average='macro'),4))}"
prec_score = f"Precision score for Decision trees Classifier is: {(round(precision_score(y_test, trees_pred, average='macro'),4))}"
acc_score = f"Accuracy score for Decision trees Classifier is: {(round(accuracy_score(y_test, trees_pred),4))}"
crs_val_score = f"Mean of cross val score for Decision trees Classifier is: {(round(np.mean(cross_val_score(clf, X, y, cv=5)),4))}"

with open("../results/decision_trees_results.npy", "wb") as f:
    np.save(f, f1)
    np.save(f, prec_score)
    np.save(f, acc_score)
    np.save(f, crs_val_score)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
vec = []

for (train_index, test_index) in rskf.split(X, y):
    x_train_2, x_test_2 = X[train_index], X[test_index]
    y_train_2, y_test_2 = y[train_index], y[test_index]
    clf.fit(x_train_2, y_train_2)
    support_matrix = clf.predict(x_test_2)
    acc_score = accuracy_score(y_test_2, support_matrix)
    vec.append(acc_score)