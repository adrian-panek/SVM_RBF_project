import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)

clf = DecisionTreeClassifier()
clf.fit(X,y)
trees_pred = clf.predict(X)

print(f"F1 score score for Scikit-learn implemented decistion trees classifier is: {f1_score(y, trees_pred, average='macro')}")
print(f"Precision score for Scikit-learn implemented decistion trees classifier is: {precision_score(y, trees_pred, average='macro')}")
print(f"Accuracy score for Scikit-learn implemented decistion trees classifier is: {accuracy_score(y, trees_pred)}")
print(f"Mean of cross val score for Scikit-learn Implemented Classifier is: {np.mean(cross_val_score(clf, X, y, cv=5))}")
