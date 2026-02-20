from sklearn.neighbors import KDTree
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

class KNN_Classification:
    def __init__(self, n_neighbors, use_kd_tree=False):
        self.n_neighbors = n_neighbors
        self.use_kd_tree = use_kd_tree
        self.X_train = None
        self.y_train = None
        self.kd_tree = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.use_kd_tree:
            self.kd_tree = KDTree(X)

    def predict(self, X):
        if self.use_kd_tree:
            _, indices = self.kd_tree.query(X, k=self.n_neighbors)
            y_pred = np.array([np.bincount(self.y_train[indices[i]]).argmax() for i in range(X.shape[0])])
        else:
            y_pred = np.empty(X.shape[0])
            for i, x in enumerate(X):
                distances = np.linalg.norm(self.X_train - x, axis=1)
                nearest_indices = np.argsort(distances)[:self.n_neighbors]
                y_pred[i] = np.bincount(self.y_train[nearest_indices]).argmax()
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) * 100 


class KNN_Regression:
    def __init__(self, n_neighbors=1, use_kd_tree=False):
        self.n_neighbors = n_neighbors
        self.use_kd_tree = use_kd_tree
        self.X_train = None
        self.y_train = None
        self.kd_tree = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.use_kd_tree:
            self.kd_tree = KDTree(X)

    def predict(self, X):
        if self.use_kd_tree:
            _, indices = self.kd_tree.query(X, k=self.n_neighbors)
            y_pred = np.array([np.mean(self.y_train[indices[i]]) for i in range(X.shape[0])])
        else:
            y_pred = np.empty(X.shape[0])
            for i, x in enumerate(X):
                distances = np.linalg.norm(self.X_train - x, axis=1)
                nearest_indices = np.argsort(distances)[:self.n_neighbors]
                y_pred[i] = np.mean(self.y_train[nearest_indices])
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)  


def zad_3():
    #1.
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=3)
    #2. Utworzenie instancji klasyfikatora k-nn
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    #2. Dopasowanie klasyfikatora do danych
    knn_classifier.fit(X, y)
    #3. Wizualizacja granicy separacji
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary Visualization')
    plt.show()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 4.

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    # Podziel dane na zbiór treningowy i testowy
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

    # Klasyfikacja danych iris
    knn_classifier_iris = KNeighborsClassifier(n_neighbors=3)
    knn_classifier_iris.fit(X_train_iris, y_train_iris)
    y_pred_iris = knn_classifier_iris.predict(X_test_iris)
    accuracy_iris = knn_classifier_iris.score(X_test_iris, y_test_iris)

    print("Predicted labels:")
    print(y_pred_iris)
    print("Accuracy:", accuracy_iris)

    # Zwizualizuj dane przy pomocy metody PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_iris)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_pca, y_iris)

    # Generowanie regularnej siatki punktów w przestrzeni 2D
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Punkty opisujące siatkę konwertowane do "oryginalnej" przestrzeni (4D) przy pomocy metody pca.inverse_transform
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_pca = pca.inverse_transform(grid_points)

    # Wynik punktów poprzedniego kroku podawany do funkcji predict
    Z = knn_classifier.predict(grid_points)

    # Wizualizacja granicy separacji
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_iris, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Decision Boundary Visualization using PCA')
    plt.show()

    # 6. Testowanie algorytmu dla różnych wartości parametru k za pomocą kroswalidacji leave-one-out

    k_values = [1, 3, 5, 7, 9]
    scores = []
    for k in k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn_classifier, X, y, cv=LeaveOneOut(), scoring='accuracy').mean()
        scores.append(score)

    print("Results for different values of k:")
    print("k\tAccuracy")
    for i in range(len(k_values)):
        print("{}\t{:.2f}".format(k_values[i], scores[i]))

zad_3()

def zad_4():

    X, y = make_regression(n_samples=100, n_features=1, noise=5, random_state=42)

    knn_regressor = KNN_Regression(n_neighbors=3)
    knn_regressor.fit(X, y)
    y_pred = knn_regressor.predict(X)

    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, y_pred, color='red', linewidth=1, label='Regression line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Regression using k-nn')
    plt.legend()
    plt.show()
    mse = mean_squared_error(y, y_pred)
    print("Mean Squared Error:", mse)
zad_4()

