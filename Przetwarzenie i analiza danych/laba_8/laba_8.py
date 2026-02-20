import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def distp(X, C, e):
    distances = np.zeros((X.shape[0], C.shape[0]))
    for i in range(X.shape[0]):
        for j in range(C.shape[0]):
            diff = X[i] - C[j]
            distances[i, j] = np.sqrt(np.dot(diff, diff.T))
    return distances

def assign_clusters(X, C):
    distances = distp(X, C, None)
    return np.argmin(distances, axis=1)

def compute_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(X[clusters == i], axis=0)
    return centroids

def k_means(X, k, max_iter=100):

    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        clusters = assign_clusters(X, centroids)
        
        new_centroids = compute_centroids(X, clusters, k)
        
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return centroids, clusters

#autos_data = pd.read_csv('autos.csv')
#autos_data.dropna(inplace=True)
#X = autos_data[['normalized-losses', 'length', 'curb-weight', 'engine-size',
#               'horsepower', 'city-mpg', 'highway-mpg', 'price']].values

X = pd.read_csv("autos.csv")
X = X.select_dtypes(include=np.number)
X = X.dropna(axis=1)
X = X.drop(X.columns[0], axis=1)
X = X.values;

k = 3
centroids, clusters = k_means(X, k)

print("Centroids:")
print(centroids)

print("Cluster Membership:")
print(clusters)

plt.figure(figsize=(10, 6))

for i in range(k):
    plt.scatter(X[clusters == i, 2], X[clusters == i, 4], label=f'Cluster {i+1}', alpha=0.7)

plt.scatter(centroids[:, 2], centroids[:, 4], c='black', marker='x', s=200, label='Centroids')

plt.title('Clustering Results')
plt.xlabel('Curb Weight')
plt.ylabel('Horsepower')
plt.legend()
plt.grid(True)
plt.show()
def clustering_quality(X, centroids, clusters):
    K = len(centroids)
    total_intra_distance = 0
    total_inter_distance = 0

    for i in range(K):
        for j in range(i + 1, K):
            total_inter_distance += np.linalg.norm(centroids[i] - centroids[j]) ** 2

    for i, centroid in enumerate(centroids):
        cluster_points = X[clusters == i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        total_intra_distance += np.sum(distances ** 2)

    quality = total_intra_distance / total_inter_distance
    return quality

clustering_quality_score = clustering_quality(X, centroids, clusters)
print("Clustering Quality Score:", clustering_quality_score)
