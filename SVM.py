import numpy as np
from scipy.spatial.distance import cdist
from pandas import read_csv
from sklearn.datasets import make_blobs, make_classification, make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

class SVM(BaseEstimator):
    def __init__(self, learning_rate, la, n_iters):
        self.learning_rate = learning_rate
        self.la = la
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_pos_neg = [-1 if x<=0 else 1 for x in y]
        
        #inicjalizacja wektorów
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iters):
            i=0
            for val in X:
                #przypadek, gdy badana próbka znajduje się w dobrej klasie
                if (y_pos_neg[i] * (np.dot(val, self.w) - self.b ) >= 1 ):
                    #aktualizacja w
                    self.w = self.w-self.learning_rate * (2 * self.la * self.w)
                    #w tym przypadku b pozostaje niezmienione
                #przypadek, gdy próbka znajduje się w złej klasie
                else:
                    self.w = self.w - self.learning_rate * ((2 * self.la * self.w) - np.dot(y_pos_neg[i], val))
                    self.b = self.b - (self.learning_rate * y_pos_neg[i])
                i+=1

    def predict(self, X):
        y_pred = []
        for x_i in X:
            y_i = np.dot(x_i, self.w) - self.b
            y_i = 1 if y_i >=0 else 0
            y_pred.append(y_i)
        return y_pred
     
def RBF(X, gamma):
    if gamma == None:
        print("Gamma cannot be set to none!")
    dist_cdist = cdist(X, X, 'euclidean')
    K = np.exp(-gamma*(dist_cdist)**2)
    return K


