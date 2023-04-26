import numpy as np
from scipy.spatial.distance import cdist
import ipdb
from pandas import read_csv
from sklearn.datasets import make_blobs, make_classification, make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel

class SVM(BaseEstimator):
    def __init__(self):
        self.learning_rate = 0.001
        self.la = 0.01
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_pos_neg = [-1 if x<=0 else 1 for x in y]
        print(y_pos_neg)
        
        #inicjalizacja wektorów
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(50):
            i=0
            for val in X:
                #przypadek, gdy badana próbka znajduje się w dobrej klasie
                if (y_pos_neg[i] * (np.dot(X[i], self.w) - self.b ) >= 1 ):
                    #aktualizacja w
                    self.w = self.w-self.learning_rate * (2 * self.la * self.w)
                    #w tym przypadku b pozostaje niezmienione
                #przypadek, gdy próbka znajduje się w złej klasie
                else:
                    self.w = self.w - self.learning_rate * ((2 * self.la * self.w) - np.dot(y_pos_neg[i], X[i]))
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

X, y = make_circles(n_samples=500, noise=0.06, random_state=58)
# X = RBF(X, 1.0)
# X = rbf_kernel(X, gamma=1.0)

# testy zaimplementowanego klasyfikatora
MyClassifier = SVM()
MyClassifier.fit(X,y)
# MyClass_pred = MyClassifier.predict(X)
# print(f"My implemented Classifier prediction: {MyClass_pred}")
# print(f"My implemented Classifier accuracy: {accuracy_score(y, MyClass_pred)}")

# ImplementedClassifier = SVC()
# ImplementedClassifier.fit(X,y)
# ImplementedClass_pred = ImplementedClassifier.predict(X)
# print(f"Scikit-learn implemented Classifier prediction: {ImplementedClass_pred}")
# print(f"Scikit-learn implemented Classifier accuracy: {accuracy_score(y, ImplementedClass_pred)}")



# dataset = read_csv("dataset/Cancer_Data.csv")
# y = []
# X = [[rad_mn, txt_mn] for rad_mn, txt_mn in zip(dataset.radius_mean, dataset.texture_mean)]

# for val in dataset.diagnosis:
#     if val == 'M':
#         y.append(1)
#     y.append(0)