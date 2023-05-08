from sklearn.svm import SVC

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)

ImplementedClassifier = SVC()
ImplementedClassifier.fit(X,y)
ImplementedClass_pred = ImplementedClassifier.predict(X)

print(f"F1 score score for Scikit-learn Implemented Classifier is: {f1_score(y, ImplementedClass_pred, average='macro')}")
print(f"Precision score for Scikit-learn Implemented Classifier is: {precision_score(y, ImplementedClass_pred, average='macro')}")
print(f"Accuracy score for Scikit-learn Implemented Classifier is: {accuracy_score(y, ImplementedClass_pred)}")
