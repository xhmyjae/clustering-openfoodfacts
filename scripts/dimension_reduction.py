# scripts/dimension_reduction.py

import numpy as np
from typing import Optional, Dict
import yaml
import os
from umap import UMAP
from sklearn.preprocessing import StandardScaler

def load_params() -> Dict:
    """
    Load UMAP parameters from YAML file.
    
    Returns:
    --------
    Dict containing the parameters
    """
    yaml_path = os.path.join('data', 'clustering_params.yaml')
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)
    return params['umap']

def reduce_dimensions(
    data: np.ndarray,
    params: Optional[Dict] = None,
    scale_data: bool = True
) -> np.ndarray:
    """
    Reduce dimensionality of data using UMAP.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to reduce
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    scale_data : bool, default=True
        Whether to scale the data before reduction
    
    Returns:
    --------
    np.ndarray
        Reduced dimensionality data
    """
    # Load parameters if not provided
    if params is None:
        params = load_params()
    
    # Scale the data if requested
    if scale_data:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
    
    # Initialize and fit UMAP
    umap = UMAP(
        n_neighbors=params.get('n_neighbors', 15),
        min_dist=params.get('min_dist', 0.1),
        n_components=params.get('n_components', 2),
        random_state=params.get('random_state', 42),
        metric=params.get('metric', 'euclidean')
    )
    
    # Fit and transform the data
    reduced_data = umap.fit_transform(data_scaled)
    
    return reduced_data