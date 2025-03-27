import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import yaml
import os
from . import plots
from . import model_utils

def load_params() -> Dict:
    """
    Load clustering parameters from YAML file.
    
    Returns:
    --------
    Dict containing the parameters
    """
    yaml_path = os.path.join('data', 'clustering_params.yaml')
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)
    return params['kmeans']

def kmeans_clustering(
    data: np.ndarray,
    params: Optional[Dict] = None,
    return_metrics: bool = True
) -> Tuple[KMeans, np.ndarray, Optional[dict]]:
    """
    Perform K-means clustering on the input data with parameters from YAML file.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to cluster
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    return_metrics : bool, default=True
        Whether to return additional clustering metrics
    
    Returns:
    --------
    Tuple containing:
    - KMeans object
    - Cluster labels
    - Dictionary of metrics (if return_metrics=True)
    """
    # Load parameters if not provided
    if params is None:
        params = load_params()
    
    # Scale the data if requested
    if params.get('scale_data', True):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
    
    # Initialize and fit KMeans
    kmeans = KMeans(
        n_clusters=params['n_clusters'],
        init=params['init'],
        n_init=params['n_init'],
        max_iter=params['max_iter'],
        tol=params['tol'],
        random_state=params['random_state']
    )
    
    # Fit the model and get predictions
    labels = kmeans.fit_predict(data_scaled)
    
    # Calculate metrics if requested
    metrics = None
    if return_metrics:
        metrics = {
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_,
            'cluster_centers': kmeans.cluster_centers_,
            'labels': labels
        }
    
    # Save the model if requested
    if params.get('save_model', False):
        model_utils.save_model(
            kmeans,
            params['model_name'],
            params['model_dir']
        )
    
    return kmeans, labels, metrics

def plot_elbow_method(
    data: np.ndarray,
    params: Optional[Dict] = None
) -> None:
    """
    Plot the elbow method to help determine the optimal number of clusters.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    """
    # Load parameters if not provided
    if params is None:
        params = load_params()
    
    elbow_params = params['elbow']
    inertias = []
    K = range(1, elbow_params['max_clusters'] + 1)
    
    for k in K:
        kmeans = KMeans(
            n_clusters=k,
            init=elbow_params['init'],
            n_init=elbow_params['n_init'],
            random_state=elbow_params['random_state']
        )
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Use the plotting function from plots.py
    plots.plot_elbow(inertias)
