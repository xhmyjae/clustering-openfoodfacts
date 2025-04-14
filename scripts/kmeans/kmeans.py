import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def kmeans_clustering(data, n_clusters=8, random_state=42):
    """
    Perform K-means clustering on the given data.
    
    Parameters:
    -----------
    data : array-like
        The data to cluster
    n_clusters : int, default=8
        Number of clusters to form
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    kmeans : KMeans
        Fitted KMeans model
    labels : array
        Cluster labels for each data point
    metrics : dict
        Dictionary containing clustering metrics
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)
    
    # Calculate metrics
    metrics = {
        'inertia': kmeans.inertia_,
        'n_iter': kmeans.n_iter_,
        'silhouette': silhouette_score(data, labels)
    }
    
    return kmeans, labels, metrics

def plot_elbow_method(data, k_range=range(1, 11), random_state=42):
    """
    Plot the elbow method curve for K-means clustering.
    
    Parameters:
    -----------
    data : array-like
        The data to analyze
    k_range : range, default=range(1, 11)
        Range of k values to test
    random_state : int, default=42
        Random state for reproducibility
    """
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return k_range, inertias

def find_elbow_point(data, k_range=range(1, 11), random_state=42):
    """
    Automatically detect the elbow point using the KneeLocator.
    
    Parameters:
    -----------
    data : array-like
        The data to analyze
    k_range : range, default=range(1, 11)
        Range of k values to test
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    elbow_k : int
        The optimal number of clusters (k) at the elbow point
    """
    # Calculate inertias
    k_range, inertias = plot_elbow_method(data, k_range, random_state)
    
    # Find the elbow point using KneeLocator
    kl = KneeLocator(
        list(k_range),
        inertias,
        curve='convex',
        direction='decreasing'
    )
    
    elbow_k = kl.elbow
    
    if elbow_k is not None:
        print(f"Optimal number of clusters (k) at elbow point: {elbow_k}")
        
        # Plot the elbow point
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.plot(elbow_k, inertias[elbow_k-1], 'ro', markersize=10, label='Elbow Point')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method with Detected Elbow Point')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("No clear elbow point detected. Consider using a different range of k values.")
    
    return elbow_k

def analyze_clusters(data, labels, feature_names=None):
    """
    Analyze and visualize the characteristics of each cluster.
    
    Parameters:
    -----------
    data : array-like
        The original data
    labels : array
        Cluster labels for each data point
    feature_names : list, optional
        Names of the features for better visualization
    """
    n_clusters = len(np.unique(labels))
    
    # Calculate mean values for each cluster
    cluster_means = []
    for i in range(n_clusters):
        cluster_data = data[labels == i]
        cluster_means.append(np.mean(cluster_data, axis=0))
    
    cluster_means = np.array(cluster_means)
    
    # Plot cluster characteristics
    plt.figure(figsize=(12, 8))
    plt.imshow(cluster_means, aspect='auto', cmap='viridis')
    plt.colorbar(label='Mean Value')
    plt.xlabel('Features')
    plt.ylabel('Cluster')
    plt.title('Cluster Characteristics')
    
    if feature_names is not None:
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Print cluster sizes
    print("\nCluster sizes:")
    for i in range(n_clusters):
        print(f"Cluster {i}: {np.sum(labels == i)} samples")
    
    return cluster_means
