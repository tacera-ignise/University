from sklearn.datasets import load_digits
from sklearn import datasets 
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib.pyplot as plt

def wiPCA(data, target_dim):

    mean_vector = np.mean(data, axis=0)
    
    covariance_matrix = np.cov(data.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    selected_eigenvectors = sorted_eigenvectors[:, :target_dim]
    
    transformed_data = np.dot(data - mean_vector, selected_eigenvectors)
    
    return transformed_data, selected_eigenvectors, sorted_eigenvalues, mean_vector

rng = np.random.RandomState(1)
data = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

def zad_1():
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], label='Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    transformed_data, eigenvectors, sorted_eigenvalues, mean_vector = wiPCA(data, 1)

    plt.figure(figsize=(12, 6))

    plt.scatter(data[:, 0], data[:, 1], label='Original Data')
    plt.quiver(np.mean(data[:, 0]), np.mean(data[:, 1]), eigenvectors[0, :], eigenvectors[1, :], scale=3, color='red')

    plt.scatter(transformed_data, transformed_data, color='blue', label='Projected Data')
    print( mean_vector,eigenvectors)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

zad_1()

def zad_2():
    iris = load_iris()
    data = iris.data
    target = iris.target

    transformed_data, _, _, _ = wiPCA(data, 2)

    plt.figure(figsize=(8, 6))
    for label in np.unique(target):
        plt.scatter(transformed_data[target == label, 0], transformed_data[target == label, 1], label=f'Class {label}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('PCA on Iris data')
    plt.legend()
    plt.grid(True)
    plt.show()

zad_2()

def zad_3():
    digits = load_digits()
    data = digits.data
    target = digits.target

    transformed_data, _, eigenvalues, _ = wiPCA(data, 2)

    variance_explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    plt.plot(range(1, len(variance_explained) + 1), variance_explained, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.title('Variance Explained by Components')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    for label in np.unique(target):
        plt.scatter(transformed_data[target == label, 0], transformed_data[target == label, 1], label=f'Class {label}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('PCA on digits data')
    plt.legend()
    plt.grid(True)
    plt.show()

zad_3()
















