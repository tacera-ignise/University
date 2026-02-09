import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NBCDiscrete(BaseEstimator, ClassifierMixin):
    def __init__(self, domain_sizes, laplace=False, logs=False):
        self.domain_sizes = domain_sizes
        self.laplace = laplace
        self.logs = logs
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = self.classes_.size
        m, n = X.shape
        q_max = np.max(self.domain_sizes)
        self.PY_ = np.zeros(K)
        self.P_ = np.zeros((K, n, q_max))
        yy = np.zeros(m, dtype=np.int8)
        for y_index, label in enumerate(self.classes_):
            indexes = y == label
            self.PY_[y_index] = np.mean(indexes)
            yy[indexes] = y_index
        for i in range(m):
            for j in range(n):
                self.P_[yy[i], j, X[i, j]] += 1        
        for y_index in range(K):
            if not self.laplace:
                self.P_[y_index] /= self.PY_[y_index] * m
            else:
                for j in range(n):
                    self.P_[y_index, j] = (self.P_[y_index, j] + 1) / (self.PY_[y_index] * m + self.domain_sizes[j]) 
        if self.logs:
            self.P_ = np.log(self.P_ + 1e-10)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    
    def predict_proba(self, X):
        m, n = X.shape
        K = self.classes_.size
        probas = np.ones((m, K))
        for i in range(m):
            for y_index in range(K):
                for j in range(n):
                    probas[i, y_index] += self.P_[y_index, j, X[i, j]]
                probas[i, y_index] += np.log(self.PY_[y_index] + 1e-10)
        return probas
