import numpy as np

from SVM import SVM, RBF

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)
X = RBF(X, 0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

MyClassifier = SVM(0.001, 0.01, 100)
MyClassifier.fit(X_train, y_train)
MyClass_pred = MyClassifier.predict(X_test)

print(f"F1 score score for Scikit-learn Implemented Classifier is: {(round(f1_score(y_test, MyClass_pred, average='macro'),4))}")
print(f"Precision score for Scikit-learn Implemented Classifier is: {(round(precision_score(y_test, MyClass_pred, average='macro'),4))}")
print(f"Accuracy score for Scikit-learn Implemented Classifier is: {(round(accuracy_score(y_test, MyClass_pred),4))}")
print(f"Mean of cross val score for Scikit-learn Implemented Classifier is: {(round(np.mean(cross_val_score(MyClassifier, X, y, cv=5, scoring='accuracy')),4))}")
