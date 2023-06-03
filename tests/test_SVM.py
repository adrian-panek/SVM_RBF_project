import pytest
import numpy

from SVM import SVM, RBF

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, random_state=59)
clf = SVM(0.1, 1, 100)
clf.fit(X, y)
prediction = clf.predict(X)

with open('../results/metrics/DecisionTreeClassifier()_results.npy', 'rb') as f:
    a = numpy.load(f)

def testPredictionNotEmpty():
    assert len(prediction) != 0

def testPredictionHasCorrectValues():
    for i in range(len(prediction)):
        assert prediction[i] == 1 or prediction[i] == 0

def testResultsAreNotEmpty():
    assert len(a) != 0

def testResultsAreCorrectType():
    for val in a:
        assert type(val) == numpy.float64

def testKernelIsCorrect():
    X_kerneled = RBF(X, 2.0)
    assert X.shape != X_kerneled.shape