import pytest

from SVM import SVM

from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05)

clf = SVM()
clf.fit(x, y)
prediction = clf.predict(x[1])
assert prediction == 1 or prediction == -1