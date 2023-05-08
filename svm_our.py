from SVM import SVM

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, accuracy_score

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=59)

MyClassifier = SVM(0.001, 0.01, 100)
MyClassifier.fit(X,y)
MyClass_pred = MyClassifier.predict(X)

print(f"F1 score score for My Implemented Classifier is: {f1_score(y, MyClass_pred, average='macro')}")
print(f"Precision score for My Implemented Classifier is: {precision_score(y, MyClass_pred, average='macro')}")
print(f"Accuracy score for My Implemented Classifier is: {accuracy_score(y, MyClass_pred)}")