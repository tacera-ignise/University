import numpy as np
from klasa1 import NBCDiscrete

def read_wine_data(path):
    D = np.genfromtxt(path, delimiter=",")
    X = D[:, 1:]
    y = D[:, 0].astype(np.int8)
    return X, y

def train_test_split(X, y, train_ratio=0.75):
    m = X.shape[0]
    indexes = np.random.permutation(m)
    X = X[indexes]
    y = y[indexes]
    i = int(np.round(train_ratio * m))
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:]
    y_test = y[i:]
    return X_train, y_train, X_test, y_test

def discretize(X, bins, mins_ref=None, maxes_ref=None):
    if mins_ref is None:
        mins_ref = np.min(X, axis=0)
        maxes_ref = np.max(X, axis=0)
    X_d = np.clip(np.int8((X - mins_ref) / (maxes_ref - mins_ref) * bins), 0, bins - 1)
    return X_d, mins_ref, maxes_ref

if __name__ == "__main__":
    np.random.seed(1)
    bins = 5
    X, y = read_wine_data("wine.data")
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.75)
    X_train_d, mins_ref, maxes_ref = discretize(X_train, bins)
    X_test_d, _, _ = discretize(X_test, bins, mins_ref, maxes_ref)
    n = X.shape[1]
    domain_sizes = np.ones(n, dtype=np.int8) * bins # [5, 5, ..., 5]     
    clf = NBCDiscrete(domain_sizes,laplace=True)
    clf.fit(X_train_d, y_train)
    acc_train=clf.score(X_train_d,y_train)
    acc_test=clf.score(X_test_d,y_test)
    print(f"ACCs -> TRAIN: {acc_train}, TEST: {acc_test}")
    
    
    
    
    #print(y_test)
    #y_pred=clf.predict(X_test_d)
    #print(y_pred)
   # print(y_pred==y_test)
    #print(np.mean(y_pred==y_test))
    
    