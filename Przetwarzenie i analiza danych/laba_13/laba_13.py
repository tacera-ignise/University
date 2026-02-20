from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.stats import mode
import scipy
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import cv2


# 1. Wczytanie danych i rozdzielenie na X i Y
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# 2. Klasteryzacja metodami aglomeracyjnymi
linkage_types = ['single', 'average', 'complete', 'ward']
clusters = {}
for link_type in linkage_types:
    clustering = AgglomerativeClustering(n_clusters=3, linkage=link_type)
    clusters[link_type] = clustering.fit_predict(X)

# 3. Dopasовка результатов кластеризации к реальным классам
def findperm(n_clusters, Y_real, Y_pred):
    perm = []
    for i in range(n_clusters):
        idx = Y_pred == i
        new_label = scipy.stats.mode(Y_real[idx])[0]
        perm.append(new_label)
    return np.array([perm[label] for label in Y_pred])

mapped_clusters = {}
for link_type in linkage_types:
    mapped_clusters[link_type] = findperm(3, Y, clusters[link_type])
# 4. Obliczenie indeksu Jaccarda
jaccard_scores = {}
for linkage in linkage_types:
    jaccard_scores[linkage] = jaccard_score(Y, mapped_clusters[linkage], average='macro')

print("Indeksy Jaccarda:", jaccard_scores)

# 5. Wizualizacja w przestrzeni 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

def plot_clusters(X, y, title):
    plt.figure()
    unique_labels = np.unique(y)
    for label in unique_labels:
        cluster_points = X[y == label]
        if len(cluster_points) < 3:
            continue  # Skip plotting if there are less than 3 points in the cluster
        hull = ConvexHull(cluster_points)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-')
    plt.title(title)
    plt.legend()
    plt.show()

plot_clusters(X_reduced, Y, "Rzeczywiste etykiety")
for linkage2 in linkage_types:
    plot_clusters(X_reduced, mapped_clusters[linkage2], f"Etykiety klasteryzacji - metoda {linkage2}")

# 6. Wizualizacja w przestrzeni 3D
pca_3d = PCA(n_components=3)
X_reduced_3d = pca_3d.fit_transform(X)

def plot_clusters_3d(X, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(y)
    for label in unique_labels:
        cluster_points = X[y == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {label}')
    ax.set_title(title)
    plt.legend()
    plt.show()

plot_clusters_3d(X_reduced_3d, Y, "Rzeczywiste etykiety 3D")
for linkage1 in linkage_types:
    plot_clusters_3d(X_reduced_3d, mapped_clusters[linkage1], f"Etykiety klasteryzacji - metoda {linkage1} 3D")

# 7. Rysowanie dendrogramu
Z = scipy.cluster.hierarchy.linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=Y)
plt.title('Dendrogram (Ward)')
plt.show()



# 8. Klasteryzacja metodami K-means i GMM
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components=3)
gmm_labels = gmm.fit_predict(X)

mapped_kmeans_labels = findperm(3, Y, kmeans_labels)
mapped_gmm_labels = findperm(3, Y, gmm_labels)

jaccard_kmeans = jaccard_score(Y, mapped_kmeans_labels, average='macro')
jaccard_gmm = jaccard_score(Y, mapped_gmm_labels, average='macro')

print("Indeks Jaccarda dla K-means:", jaccard_kmeans)
print("Indeks Jaccarda dla GMM:", jaccard_gmm)
# Wizualizacja wyników K-means i GMM

plot_clusters(X_reduced, mapped_kmeans_labels, "Etykiety K-means")
plot_clusters_3d(X_reduced_3d, mapped_kmeans_labels, "Etykiety K-means 3D")
plot_clusters(X_reduced, mapped_gmm_labels, "Etykiety GMM")
plot_clusters_3d(X_reduced_3d, mapped_gmm_labels, "Etykiety GMM 3D")





image = cv2.imread('drzewo.jpg')
if image is None:
    raise ValueError("Image not found or path is incorrect")

# Zmiana rozmiaru obrazu
image = cv2.resize(image, (640, 480))

# Przekształcenie obrazu do formatu 307200x3
pixels = image.reshape((-1, 3))

# Definicja liczby klastrów i metod klastrowania
clusters = [2, 3, 5, 10]
methods = ['KMeans', 'GaussianMixture']

# Iteracja przez metody klastrowania
for method in methods:
    for n_clusters in clusters:

        # Klastrowanie
        if method == 'KMeans':
            model = KMeans(n_clusters=n_clusters, n_init=10)
        elif method == 'GaussianMixture':
            model = GaussianMixture(n_components=n_clusters)
        
        else:
            raise ValueError("Invalid method") 
        
        labels = model.fit_predict(pixels)

        # Obliczenie środków klastrów
        if method == 'KMeans':
            centers = model.cluster_centers_
        elif method == 'GaussianMixture':
            centers = model.means_
        elif method == 'Agglomerative':
            centers = np.array([pixels[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Kwantyzacja wektorowa
        quantized_pixels = centers[labels].astype(np.uint8)
        quantized_image = quantized_pixels.reshape((480, 640, 3))

        # Obliczenie błędu kwantyzacji
        error = mean_squared_error(image.reshape((-1, 3)), quantized_pixels)
        error_image = ((image - quantized_image) ** 2).mean(axis=2)

        # Wizualizacja obrazów i błędu
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Quantized Image\n{method} with {n_clusters} Clusters\nQuantization Error: {error:.4f}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(error_image, cmap='gray')
        plt.title('Quantization Error')
        plt.axis('off')
        plt.show()
        original_shape = image.shape
        inverse_quantized_image = centers[labels].reshape(original_shape)   

        # Wizualizacja rozkładu błędu dla konkretnych pikseli
        plt.figure(figsize=(6, 6))
        plt.imshow(error_image, cmap='gray')
        plt.title('Distribution of Quantization Error')
        plt.colorbar()
        plt.axis('off')

        plt.show()