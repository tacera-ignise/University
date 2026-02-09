
import numpy as np
from class1 import NBCDiscrete
import pandas as pd

def read_data(path):
    df = pd.read_csv(path, delimiter=";")
    df = df.dropna()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = pd.factorize(df[column])[0]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def discretize(X, bins, mins_ref=None, maxes_ref=None):
    if mins_ref is None:
        mins_ref = np.min(X, axis=0)
        maxes_ref = np.max(X, axis=0)
    X_d = np.clip(np.int8((X - mins_ref) / (maxes_ref - mins_ref) * bins), 0, bins - 1)
    return X_d, mins_ref, maxes_ref


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




if __name__ == "__main__":
    data_path = "data.csv"  
    print("Laplace=True, Logs=True")
    experiment(data_path, bins_range=(5, 8), laplace=True, logs=True)
    print("\nLaplace=False, Logs=True")
    experiment(data_path, bins_range=(5, 8), laplace=False, logs=True)
    print("\nLaplace=True, Logs=False")
    experiment(data_path, bins_range=(5, 8), laplace=True, logs=False)
    print("\nLaplace=False, Logs=False")
    experiment(data_path, bins_range=(5, 8), laplace=False, logs=False)
    test_numerical_safety()
    

def experiment(data_path, bins_range=(5, 20), laplace=False, logs=False):
    X, y = read_data(data_path)
    results = []
    
    for bins in range(bins_range[0], bins_range[1]+1):
        X_train, y_train, X_test, y_test = train_test_split(X, y)
        X_train_d, mins_ref, maxes_ref = discretize(X_train, bins)
        X_test_d, _, _ = discretize(X_test, bins, mins_ref, maxes_ref)
        
        domain_sizes = np.ones(X.shape[1], dtype=np.int8) * bins
        clf = NBCDiscrete(domain_sizes, laplace=laplace, logs=logs)
        clf.fit(X_train_d, y_train)
        
        acc_train = clf.score(X_train_d, y_train)
        acc_test = clf.score(X_test_d, y_test)
        results.append((bins, acc_train, acc_test))
        
        print(f"Bins: {bins}, Train Accuracy: {acc_train:.4f}, Test Accuracy: {acc_test:.4f}")
    
    return results
def test_numerical_safety():
    X, y = read_data("data.csv")
    X = np.hstack([X] * 10) 
    bins = 5
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train_d, mins_ref, maxes_ref = discretize(X_train, bins)
    X_test_d, _, _ = discretize(X_test, bins, mins_ref, maxes_ref)
    
    domain_sizes = np.ones(X.shape[1], dtype=np.int8) * bins
    clf = NBCDiscrete(domain_sizes, laplace=True, logs=True)
    clf.fit(X_train_d, y_train)
    
    acc_train = clf.score(X_train_d, y_train)
    acc_test = clf.score(X_test_d, y_test)
    
    print(f"Train Accuracy with increased features: {acc_train:.4f}")
    print(f"Test Accuracy with increased features: {acc_test:.4f}")