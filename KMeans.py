import numpy as np
import matplotlib as plt

"""
Kmeans clustering with addition KMeans++ initialization
Control the initialization by parameter 'pp' in method 'fit' (pp=1 implies KMeans++)
"""

class KMeans:
    def __init__(self):
        self.k = None
        self.assignment = None
        self.centroids = None

    def distance(self, X, centroid, type='euclid'):
        if type == 'euclid':
            dist = np.linalg.norm(X - centroid, axis=0)
            return dist

    def converge(self, new_centroids, centroids, new_assignment, assignment, new_error, error, epsilon):
        if all(self.distance(new_centroids, centroids)) < epsilon:
            return True
        if np.sum(abs(new_assignment - assignment)) < 5:
            return True
        if abs(new_error - error) < epsilon:
            return True
        return False
    
    def error(self, centroids, X, assignment):
        error = 0
        for i in range(self.k):
            centroid_data = X[:, assignment == i]
            if centroid_data.size > 0:
                d = self.distance(centroid_data, centroids[:, i].reshape(X.shape[0], 1))**2
                error += np.sum(d)
        return error

    def fit(self, X, k=3, max_ite=100, type='euclid', epsilon=1e-6, pp=0):
        self.k = k
        error = 0
        n_feature, n_example = X.shape
        
        if pp == 0:
            random_indices = np.random.choice(n_example, k, replace=False)
            centroids = X[:, random_indices]
        elif pp == 1:
            random_index = np.random.choice(n_example)
            centroids = np.zeros((n_feature, k))
            centroids[:, 0] = X[:, random_index]
            used = [-1 for _ in range(k)]
            used[0] = random_index
            for i in range(1,k):
                c = None
                max_d = -1e10
                for j in range(n_example):
                    if j in used:
                        continue
                    x = X[:, j]
                    d = self.distance(x.reshape(n_feature, 1), centroids[:, 0:i])
                    min_d = np.min(d)
                    if min_d > max_d:
                        c = x
                        max_d = min_d
                        used[i] = j
                centroids[:, i] = c
            plot_data_and_centroids(X, centroids)


        assignment = np.array([-1 for _ in range(n_example)]).reshape(1, n_example)
    
        for i in range(max_ite):
            print(f'Epoch {i}')
            # 1. Assign each point to the nearest centroid
            dist = np.zeros((k, n_example))
            for j in range(k):
                centroid = centroids[:, j].reshape(n_feature, 1)
                dist[j] = self.distance(X, centroid, type='euclid')
            new_assignment = np.argmin(dist, axis=0)
            # 2. Update the centroid
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                centroid_data = X[:, new_assignment == j]
                if centroid_data.size > 0:
                    new_centroids[:, j] = np.mean(centroid_data, axis=1)
                else:
                    new_centroids[:, j] = centroids[:, j]
            # 3. Check the convergence
            new_error = self.error(new_centroids, X, new_assignment)
            if self.converge(new_centroids, centroids, new_assignment, assignment, new_error, error, epsilon) or i == max_ite - 1:
                self.assignment = new_assignment
                self.centroids = new_centroids
                break
            # 4. If not converged, repeat the process
            centroids = new_centroids
            assignment = new_assignment
            error = new_error
            # 5. If converged, break the loop

# Helper functions to visualize the result

def plot_samples(samples):
    plt.scatter(samples[:, 0], samples[:, 1], c='blue', marker='o')
    plt.title('Scatter plot of samples')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_clusters(centroids, data_clustered,mode=0):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Define a list of colors for different clusters
    plt.figure(figsize=(10, 6))
    for key in data_clustered:
        if len(data_clustered[key]) > 0:  # Check if the cluster is not empty
            points = np.hstack(data_clustered[key])
            plt.scatter(points[0, :], points[1, :], c=colors[key % len(colors)], label=f'Cluster {key}')
    plt.scatter(centroids[0, :], centroids[1, :], c='black', marker='x', s=100, label='Centroids')
    if mode == 0:
        plt.title('Clusters and Centroids')
    else:
        plt.title('Final')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_data_and_centroids(data, centroids):
    plt.figure(figsize=(10, 6))
    # Plot data points
    plt.scatter(data[0, :], data[1, :], c='blue', marker='o', label='Data Points', alpha=0.5)
    # Plot centroids
    plt.scatter(centroids[0, :], centroids[1, :], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title('Data Points and Initial Centroids (K-means++)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_kmeans_results(X, centroids, assignments):
    plt.figure(figsize=(10, 6))
    
    # Define colors for different clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # Plot points for each cluster with different colors
    for i in range(len(centroids[0])):  # Iterate over number of clusters
        # Get points assigned to this cluster
        cluster_points = X[:, assignments == i]
        if cluster_points.size > 0:  # Check if cluster is not empty
            plt.scatter(cluster_points[0, :], cluster_points[1, :], 
                       c=colors[i % len(colors)], 
                       marker='o', 
                       label=f'Cluster {i}')
    
    # Plot centroids
    plt.scatter(centroids[0, :], centroids[1, :], 
               c='black', 
               marker='x', 
               s=100, 
               linewidths=3, 
               label='Centroids')
    
    plt.title('K-means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()