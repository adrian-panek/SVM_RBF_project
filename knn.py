from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0)

neigh = KNeighborsClassifier()
neigh.fit(X,y)
trees_pred = neigh.predict(X)

print(f"F1 score score for Scikit-learn implemented decistion trees classifier is: {f1_score(y, trees_pred, average='macro')}")
print(f"Precision score for Scikit-learn implemented decistion trees classifier is: {precision_score(y, trees_pred, average='macro')}")
print(f"Accuracy score for Scikit-learn implemented decistion trees classifier is: {accuracy_score(y, trees_pred)}")