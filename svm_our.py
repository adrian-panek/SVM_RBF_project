import numpy as np

from SVM import SVM, RBF

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)
X = RBF(X, 0.5)

MyClassifier = SVM(0.001, 0.01, 100)
MyClassifier.fit(X,y)
MyClass_pred = MyClassifier.predict(X)

print(f"F1 score score for My Implemented Classifier is: {f1_score(y, MyClass_pred, average='macro')}")
print(f"Precision score for My Implemented Classifier is: {precision_score(y, MyClass_pred, average='macro')}")
print(f"Accuracy score for My Implemented Classifier is: {accuracy_score(y, MyClass_pred)}")
print(f"Mean of cross val score for Scikit-learn Implemented Classifier is: {np.mean(cross_val_score(MyClassifier, X, y, cv=5, scoring='accuracy'))}")