import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import psutil
import os

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def find_optimal_parameters(data, min_cluster_size_range=range(5, 51, 5), min_samples_range=range(5, 21, 5), 
                          max_memory_mb=1000, batch_size=10000):
    """
    Find optimal HDBSCAN parameters by grid search with memory constraints.
    
    Parameters:
    -----------
    data : array-like
        The data to cluster
    min_cluster_size_range : range
        Range of min_cluster_size values to test
    min_samples_range : range
        Range of min_samples values to test
    max_memory_mb : int
        Maximum memory usage in MB before stopping
    batch_size : int
        Size of batches for processing large datasets
        
    Returns:
    --------
    best_params : dict
        Dictionary containing the best parameters found
    """
    best_score = -1
    best_params = {}
    results = []
    
    # If data is too large, use a sample
    if len(data) > batch_size:
        print(f"Dataset too large ({len(data)} samples). Using a sample of {batch_size} samples.")
        indices = np.random.choice(len(data), batch_size, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data
    
    total_combinations = len(min_cluster_size_range) * len(min_samples_range)
    pbar = tqdm(total=total_combinations, desc="Parameter search")
    
    for min_cluster_size in min_cluster_size_range:
        for min_samples in min_samples_range:
            # Check memory usage
            current_memory = get_memory_usage()
            if current_memory > max_memory_mb:
                print(f"\nMemory usage ({current_memory:.0f}MB) exceeded limit ({max_memory_mb}MB). Stopping search.")
                break
            
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    memory=None  # Disable memory caching
                )
                labels = clusterer.fit_predict(data_sample)
                
                # Skip if all points are noise (-1)
                if len(np.unique(labels)) <= 1:
                    pbar.update(1)
                    continue
                    
                # Calculate silhouette score (excluding noise points)
                mask = labels != -1
                if np.sum(mask) > 1:  # Need at least 2 points for silhouette score
                    score = silhouette_score(data_sample[mask], labels[mask])
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'score': score,
                        'n_clusters': len(np.unique(labels[labels != -1])),
                        'noise_ratio': np.sum(labels == -1) / len(labels)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples
                        }
            except Exception as e:
                print(f"\nError during clustering: {str(e)}")
                continue
            finally:
                pbar.update(1)
    
    pbar.close()
    
    if not results:
        print("No valid results found. Try different parameter ranges.")
        return {'min_cluster_size': 5, 'min_samples': 5}  # Default values
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Plot results if we have any
    if results:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            [r['min_cluster_size'] for r in results],
            [r['min_samples'] for r in results],
            c=[r['score'] for r in results],
            cmap='viridis',
            s=[r['n_clusters'] * 50 for r in results]
        )
        
        plt.colorbar(scatter, label='Silhouette Score')
        plt.xlabel('Min Cluster Size')
        plt.ylabel('Min Samples')
        plt.title('HDBSCAN Parameter Search Results')
        plt.grid(True)
        plt.show()
        
        # Print top 5 parameter combinations
        print("\nTop 5 parameter combinations:")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. Parameters:")
            print(f"   Min Cluster Size: {result['min_cluster_size']}")
            print(f"   Min Samples: {result['min_samples']}")
            print(f"   Silhouette Score: {result['score']:.3f}")
            print(f"   Number of Clusters: {result['n_clusters']}")
            print(f"   Noise Ratio: {result['noise_ratio']:.2%}")
    
    return best_params

def hdbscan_clustering(data, min_cluster_size=None, min_samples=None, find_params=True, 
                      max_memory_mb=1000, batch_size=10000):
    """
    Perform HDBSCAN clustering on the data with memory constraints.
    
    Parameters:
    -----------
    data : array-like
        The data to cluster
    min_cluster_size : int, optional
        Minimum size of clusters
    min_samples : int, optional
        Minimum number of samples in a neighborhood
    find_params : bool, default=True
        Whether to find optimal parameters
    max_memory_mb : int
        Maximum memory usage in MB before stopping
    batch_size : int
        Size of batches for processing large datasets
        
    Returns:
    --------
    clusterer : HDBSCAN
        Fitted HDBSCAN model
    labels : array
        Cluster labels for each data point
    metrics : dict
        Dictionary containing clustering metrics
    """
    if find_params or min_cluster_size is None or min_samples is None:
        print("Finding optimal parameters...")
        params = find_optimal_parameters(data, max_memory_mb=max_memory_mb, batch_size=batch_size)
        min_cluster_size = params['min_cluster_size']
        min_samples = params['min_samples']
    
    print(f"\nPerforming HDBSCAN clustering with:")
    print(f"Min Cluster Size: {min_cluster_size}")
    print(f"Min Samples: {min_samples}")
    
    # If data is too large, use a sample
    if len(data) > batch_size:
        print(f"Dataset too large ({len(data)} samples). Using a sample of {batch_size} samples.")
        indices = np.random.choice(len(data), batch_size, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data
    
    # Perform clustering with memory constraints
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            memory=None  # Disable memory caching
        )
        labels = clusterer.fit_predict(data_sample)
        
        # Calculate metrics
        n_clusters = len(np.unique(labels[labels != -1]))
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Calculate silhouette score (excluding noise points)
        mask = labels != -1
        if np.sum(mask) > 1:
            silhouette = silhouette_score(data_sample[mask], labels[mask])
        else:
            silhouette = np.nan
        
        metrics = {
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': silhouette
        }
        
        return clusterer, labels, metrics
        
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        return None, None, None

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
    if labels is None:
        print("No labels provided for analysis.")
        return None
        
    # Get unique labels excluding noise (-1)
    unique_labels = np.unique(labels[labels != -1])
    
    if len(unique_labels) == 0:
        print("No clusters found. All points are noise.")
        return None
    
    # Calculate mean values for each cluster
    cluster_means = []
    for label in unique_labels:
        cluster_data = data[labels == label]
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
    for label in unique_labels:
        size = np.sum(labels == label)
        print(f"Cluster {label}: {size} samples")
    
    print(f"\nNoise points: {np.sum(labels == -1)} samples")
    
    return cluster_means

def plot_cluster_probabilities(clusterer, data):
    """
    Plot the probability of each point belonging to its cluster.
    
    Parameters:
    -----------
    clusterer : HDBSCAN
        Fitted HDBSCAN model
    data : array-like
        The original data
    """
    if clusterer is None:
        print("No clusterer provided for probability analysis.")
        return
        
    # Get probabilities
    probabilities = clusterer.probabilities_
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=50, alpha=0.75)
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Cluster Membership Probabilities')
    plt.grid(True)
    plt.show()
    
    # Print statistics
    print("\nProbability Statistics:")
    print(f"Mean probability: {np.mean(probabilities):.3f}")
    print(f"Median probability: {np.median(probabilities):.3f}")
    print(f"Min probability: {np.min(probabilities):.3f}")
    print(f"Max probability: {np.max(probabilities):.3f}")
