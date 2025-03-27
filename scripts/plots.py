import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
import yaml
import os

def load_params() -> Dict:
    """
    Load plotting parameters from YAML file.
    
    Returns:
    --------
    Dict containing the parameters
    """
    yaml_path = os.path.join('data', 'clustering_params.yaml')
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)
    return params['plotting']

def plot_clusters_2d(
    data: np.ndarray,
    labels: np.ndarray,
    x_idx: int = 0,
    y_idx: int = 1,
    feature_names: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    title: Optional[str] = None
) -> None:
    """
    Create a 2D scatter plot of the clusters using specified features.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    labels : np.ndarray
        Cluster labels
    x_idx : int, default=0
        Index of the feature to plot on x-axis
    y_idx : int, default=1
        Index of the feature to plot on y-axis
    feature_names : List[str], optional
        Names of the features
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    title : str, optional
        Title for the plot. If None, uses default from params
    """
    # Load parameters if not provided
    if params is None:
        params = load_params()
    
    # Create the plot
    plt.figure(figsize=tuple(params['figsize']))
    
    # Create scatter plot
    scatter = plt.scatter(data[:, x_idx], data[:, y_idx], 
                         c=labels, 
                         cmap=params['cmap'])
    
    # Set labels and title
    if feature_names is not None:
        plt.xlabel(feature_names[x_idx])
        plt.ylabel(feature_names[y_idx])
    else:
        plt.xlabel(f'Feature {x_idx}')
        plt.ylabel(f'Feature {y_idx}')
    
    plt.title(title or params['title'])
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Clusters")
    plt.gca().add_artist(legend1)
    
    plt.tight_layout()
    plt.show()

def plot_elbow(
    inertias: List[float],
    params: Optional[Dict] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot the elbow method results.
    
    Parameters:
    -----------
    inertias : List[float]
        List of inertia values for different numbers of clusters
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    title : str, optional
        Title for the plot. If None, uses default from params
    """
    # Load parameters if not provided
    if params is None:
        params = load_params()
    
    K = range(1, len(inertias) + 1)
    
    plt.figure(figsize=tuple(params['figsize']))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title(title or params['elbow_title'])
    plt.show()

def plot_cluster_sizes(
    labels: np.ndarray,
    params: Optional[Dict] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot the distribution of cluster sizes.
    
    Parameters:
    -----------
    labels : np.ndarray
        Cluster labels
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    title : str, optional
        Title for the plot. If None, uses default from params
    """
    # Load parameters if not provided
    if params is None:
        params = load_params()
    
    # Count cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=tuple(params['figsize']))
    plt.bar(unique_labels, counts)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    plt.title(title or params['size_title'])
    plt.show() 