import numpy
from sklearn.datasets import make_blobs
from sklearn.base import BaseEstimator

#learning_rate = 0.001
#lambda = 0.01


class SVM(BaseEstimator):
    def __init__(self):
        self.learning_rate = 0.001
        self.la = 0.01
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_pos_neg = [-1 if x<=0 else 1 for x in y]
        
        #inicjalizacja wektorów
        self.w = numpy.zeros(X.shape[1])
        self.b = 0

        i=0
        for val in X:
            #przypadek, gdy badana próbka znajduje się w dobrej klasie
            if (y_pos_neg[i] * (numpy.dot(x[i], self.w) - self.b ) >= 1 ):
                #aktualizacja w
                self.w = self.w-self.learning_rate * (2 * self.la * self.w)
                #w tym przypadku b pozostaje niezmienione
                print("1 case")
            else:
                print("2 case")
                self.w = self.w - self.learning_rate * ((2 * self.la * self.w) - numpy.dot(y_pos_neg[i], x[i]))
                self.b = self.b - (self.learning_rate * y_pos_neg[i])
            i+=1

    def predict(self, X):
        y = numpy.dot(X, self.w) - self.b
        return 1 if y>=0 else -1

x, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05)
cls = SVM()
cls.fit(x,y)
print(cls.predict(x[-1]))
print(y[-1])
print(f"{cls.b}, {cls.w}")
# import ipdb
# ipdb.set_trace()
