import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
import cvxopt

def RBF(X, gamma):
    dist_cdist = cdist(X, X, 'euclidean')
    K = np.exp(-gamma*(dist_cdist)**2)
    return K

class SVM(object):
    def __init__(self, kernel=RBF, la=None):
        self.kernel = kernel
        self.la = la
        if self.la is not None: self.la = float(self.la)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = self.kernel(X, 1.0)

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples)) * 1.0
        b = cvxopt.matrix(0.0)

        if self.la is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.la
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))
        
         # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel((X[i], sv), 1.0)
                y_predict[i] = s[0][0]
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
     



