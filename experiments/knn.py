import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

fig = plt.figure(figsize = (10,10))
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
plt.show()

neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)
knn_pred = neigh.predict(X_test)

f1 = f"F1 score score for Scikit-learn Implemented Classifier is: {(round(f1_score(y_test, knn_pred, average='macro'),4))}"
prec_score = f"Precision score for Scikit-learn Implemented Classifier is: {(round(precision_score(y_test, knn_pred, average='macro'),4))}"
acc_score = f"Accuracy score for Scikit-learn Implemented Classifier is: {(round(accuracy_score(y_test, knn_pred),4))}"
crs_val_score = f"Mean of cross val score for Scikit-learn Implemented Classifier is: {(round(np.mean(cross_val_score(neigh, X, y, cv=5)),4))}"

with open("../results/knn_results.npy", "wb") as f:
    np.save(f, f1)
    np.save(f, prec_score)
    np.save(f, acc_score)
    np.save(f, crs_val_score)