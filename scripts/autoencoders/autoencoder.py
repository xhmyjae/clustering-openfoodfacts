import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import yaml
import os

# Try importing matplotlib with error handling
try:
    import matplotlib
    # Use inline backend for Jupyter notebooks
    matplotlib.use('inline')
    import matplotlib.pyplot as plt
    print("Matplotlib successfully imported with inline backend")
except ImportError as e:
    print(f"Error importing matplotlib: {e}")
    raise

from .. import plots
from .. import model_utils

def load_params() -> Dict:
    """
    Load autoencoder parameters from YAML file.
    
    Returns:
    --------
    Dict containing the parameters
    """
    yaml_path = os.path.join('data', 'clustering_params.yaml')
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)
    return params.get('autoencoder', {})

def build_autoencoder(
    input_dim: int,
    encoding_dim: int = 2,
    params: Optional[Dict] = None
) -> Tuple[Model, Model, Model]:
    """
    Build an autoencoder model for dimensionality reduction.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input data
    encoding_dim : int, default=2
        Dimension of the encoded space
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    
    Returns:
    --------
    Tuple containing:
    - Full autoencoder model
    - Encoder model
    - Decoder model
    """
    if params is None:
        params = load_params()
    
    # Get layer sizes from params or use defaults
    hidden_layers = params.get('hidden_layers', [32, 16])
    
    # Input layer
    input_layer = layers.Input(shape=(input_dim,))
    
    # Encoder
    x = input_layer
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Latent space
    encoded = layers.Dense(encoding_dim, activation='linear', name='encoded')(x)
    
    # Decoder
    x = encoded
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Output layer
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    # Create models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    # Decoder model
    decoder_input = layers.Input(shape=(encoding_dim,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    decoder_output = layers.Dense(input_dim, activation='linear')(x)
    decoder = Model(decoder_input, decoder_output)
    
    return autoencoder, encoder, decoder

def autoencoder_clustering(
    data: np.ndarray,
    encoding_dim: int = 2,
    params: Optional[Dict] = None,
    return_metrics: bool = True
) -> Tuple[Model, np.ndarray, np.ndarray, Optional[dict]]:
    """
    Perform autoencoder-based dimensionality reduction and clustering.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to cluster
    encoding_dim : int, default=2
        Dimension of the encoded space
    params : Dict, optional
        Dictionary of parameters. If None, loads from YAML file
    return_metrics : bool, default=True
        Whether to return additional metrics
    
    Returns:
    --------
    Tuple containing:
    - Encoder model
    - Encoded data
    - Reconstructed data
    - Dictionary of metrics (if return_metrics=True)
    """
    if params is None:
        params = load_params()
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Build models
    autoencoder, encoder, decoder = build_autoencoder(
        input_dim=data.shape[1],
        encoding_dim=encoding_dim,
        params=params
    )
    
    # Compile autoencoder
    autoencoder.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Train autoencoder
    history = autoencoder.fit(
        data_scaled,
        data_scaled,
        epochs=params.get('epochs', 100),
        batch_size=params.get('batch_size', 32),
        validation_split=0.2,
        verbose=1
    )
    
    # Get encoded data
    encoded_data = encoder.predict(data_scaled)
    
    # Get reconstructed data
    reconstructed_data = autoencoder.predict(data_scaled)
    
    # Calculate metrics if requested
    metrics = None
    if return_metrics:
        metrics = {
            'history': history.history,
            'reconstruction_error': np.mean(np.square(data_scaled - reconstructed_data)),
            'encoded_data': encoded_data
        }
    
    # Save the model if requested
    if params.get('save_model', False):
        model_utils.save_model(
            encoder,
            params.get('model_name', 'autoencoder'),
            params.get('model_dir', 'models')
        )
    
    return encoder, encoded_data, reconstructed_data, metrics

def plot_autoencoder_results(
    encoded_data: np.ndarray,
    original_data: np.ndarray,
    reconstructed_data: np.ndarray,
    title: str = "Autoencoder Results"
) -> None:
    """
    Plot the results of autoencoder dimensionality reduction.
    
    Parameters:
    -----------
    encoded_data : np.ndarray
        Data in the encoded space
    original_data : np.ndarray
        Original input data
    reconstructed_data : np.ndarray
        Reconstructed data from autoencoder
    title : str, default="Autoencoder Results"
        Title for the plot
    """
    try:
        # Plot encoded data
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Encoded space
        plt.subplot(121)
        plt.scatter(encoded_data[:, 0], encoded_data[:, 1], alpha=0.5)
        plt.title('Encoded Space')
        plt.xlabel('First Encoded Dimension')
        plt.ylabel('Second Encoded Dimension')
        
        # Plot 2: Reconstruction error
        plt.subplot(122)
        reconstruction_error = np.mean(np.square(original_data - reconstructed_data), axis=1)
        plt.hist(reconstruction_error, bins=50)
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plotting: {e}")
        raise

def plot_autoencoder_detailed_results(
    encoded_data: np.ndarray,
    original_data: np.ndarray,
    reconstructed_data: np.ndarray,
    feature_names: Optional[list] = None,
    title: str = "Autoencoder Detailed Results"
) -> None:
    """
    Create detailed plots of autoencoder results including feature importance and reconstruction quality.
    
    Parameters:
    -----------
    encoded_data : np.ndarray
        Data in the encoded space
    original_data : np.ndarray
        Original input data
    reconstructed_data : np.ndarray
        Reconstructed data from autoencoder
    feature_names : list, optional
        Names of the features in the original data
    title : str, default="Autoencoder Detailed Results"
        Title for the plots
    """
    try:
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Encoded space scatter plot
        plt.subplot(221)
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], alpha=0.5)
        plt.title('Encoded Space')
        plt.xlabel('First Encoded Dimension')
        plt.ylabel('Second Encoded Dimension')
        plt.colorbar(scatter)
        
        # 2. Reconstruction error distribution
        plt.subplot(222)
        reconstruction_error = np.mean(np.square(original_data - reconstructed_data), axis=1)
        plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count')
        
        # 3. Feature importance (correlation with encoded dimensions)
        plt.subplot(223)
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(original_data.shape[1])]
        
        # Calculate correlations between original features and encoded dimensions
        correlations = np.zeros((len(feature_names), 2))
        for i in range(2):  # For both encoded dimensions
            for j in range(len(feature_names)):
                correlations[j, i] = np.corrcoef(original_data[:, j], encoded_data[:, i])[0, 1]
        
        # Plot feature importance
        x = np.arange(len(feature_names))
        width = 0.35
        plt.bar(x - width/2, correlations[:, 0], width, label='Encoded Dim 1')
        plt.bar(x + width/2, correlations[:, 1], width, label='Encoded Dim 2')
        plt.title('Feature Importance (Correlation with Encoded Dimensions)')
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.xticks(x, feature_names, rotation=45, ha='right')
        plt.legend()
        
        # 4. Original vs Reconstructed feature comparison
        plt.subplot(224)
        feature_idx = np.argmax(np.abs(correlations[:, 0]))  # Most important feature
        plt.scatter(original_data[:, feature_idx], reconstructed_data[:, feature_idx], 
                   alpha=0.5, c=reconstruction_error, cmap='viridis')
        plt.colorbar(label='Reconstruction Error')
        plt.title(f'Original vs Reconstructed: {feature_names[feature_idx]}')
        plt.xlabel('Original Values')
        plt.ylabel('Reconstructed Values')
        
        # Adjust layout and display
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print("\nAutoencoder Statistics:")
        print(f"Average Reconstruction Error: {np.mean(reconstruction_error):.4f}")
        print(f"Max Reconstruction Error: {np.max(reconstruction_error):.4f}")
        print(f"Min Reconstruction Error: {np.min(reconstruction_error):.4f}")
        print("\nMost Important Features:")
        for i in range(2):
            top_features = np.argsort(np.abs(correlations[:, i]))[-5:][::-1]
            print(f"\nTop 5 features for Encoded Dimension {i+1}:")
            for idx in top_features:
                print(f"{feature_names[idx]}: {correlations[idx, i]:.4f}")
                
    except Exception as e:
        print(f"Error in detailed plotting: {e}")
        raise

def plot_autoencoder_advanced_analysis(
    encoded_data: np.ndarray,
    original_data: np.ndarray,
    reconstructed_data: np.ndarray,
    feature_names: Optional[list] = None,
    title: str = "Advanced Autoencoder Analysis"
) -> None:
    """
    Create advanced visualizations for autoencoder analysis including feature distributions,
    reconstruction quality by feature, and cluster analysis.
    
    Parameters:
    -----------
    encoded_data : np.ndarray
        Data in the encoded space
    original_data : np.ndarray
        Original input data
    reconstructed_data : np.ndarray
        Reconstructed data from autoencoder
    feature_names : list, optional
        Names of the features in the original data
    title : str, default="Advanced Autoencoder Analysis"
        Title for the plots
    """
    try:
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(original_data.shape[1])]
            
        # Create a figure with multiple subplots
        plt.figure(figsize=(20, 15))
        
        # 1. Feature Distribution Comparison
        plt.subplot(331)
        feature_idx = np.argmax(np.var(original_data, axis=0))  # Most variable feature
        plt.hist(original_data[:, feature_idx], bins=50, alpha=0.5, label='Original', color='blue')
        plt.hist(reconstructed_data[:, feature_idx], bins=50, alpha=0.5, label='Reconstructed', color='red')
        plt.title(f'Distribution of Most Variable Feature: {feature_names[feature_idx]}')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend()
        
        # 2. Reconstruction Error by Feature
        plt.subplot(332)
        feature_errors = np.mean(np.square(original_data - reconstructed_data), axis=0)
        plt.bar(range(len(feature_errors)), feature_errors)
        plt.title('Average Reconstruction Error by Feature')
        plt.xlabel('Feature Index')
        plt.ylabel('Mean Squared Error')
        plt.xticks(range(len(feature_errors)), feature_names, rotation=45, ha='right')
        
        # 3. Encoded Space Density Plot
        plt.subplot(333)
        plt.hist2d(encoded_data[:, 0], encoded_data[:, 1], bins=50, cmap='viridis')
        plt.colorbar(label='Count')
        plt.title('Density Plot of Encoded Space')
        plt.xlabel('First Encoded Dimension')
        plt.ylabel('Second Encoded Dimension')
        
        # 4. Feature Correlations with Encoded Dimensions
        plt.subplot(334)
        correlations = np.zeros((len(feature_names), 2))
        for i in range(2):
            for j in range(len(feature_names)):
                correlations[j, i] = np.corrcoef(original_data[:, j], encoded_data[:, i])[0, 1]
        
        plt.imshow(correlations, aspect='auto', cmap='RdBu')
        plt.colorbar(label='Correlation')
        plt.title('Feature Correlations with Encoded Dimensions')
        plt.xlabel('Encoded Dimension')
        plt.ylabel('Feature Index')
        plt.xticks([0, 1], ['Dim 1', 'Dim 2'])
        plt.yticks(range(len(feature_names)), feature_names, rotation=0)
        
        # 5. Reconstruction Quality Scatter Plot
        plt.subplot(335)
        # Calculate reconstruction errors for the specific feature
        feature_reconstruction_errors = np.square(original_data[:, feature_idx] - reconstructed_data[:, feature_idx])
        scatter = plt.scatter(original_data[:, feature_idx], reconstructed_data[:, feature_idx], 
                            alpha=0.5, c=feature_reconstruction_errors, cmap='viridis')
        plt.colorbar(scatter, label='Reconstruction Error')
        plt.plot([original_data[:, feature_idx].min(), original_data[:, feature_idx].max()],
                [original_data[:, feature_idx].min(), original_data[:, feature_idx].max()],
                'r--', label='Perfect Reconstruction')
        plt.title(f'Reconstruction Quality: {feature_names[feature_idx]}')
        plt.xlabel('Original Values')
        plt.ylabel('Reconstructed Values')
        plt.legend()
        
        # 6. Feature Importance Heatmap
        plt.subplot(336)
        feature_importance = np.abs(correlations)
        plt.imshow(feature_importance, aspect='auto', cmap='YlOrRd')
        plt.colorbar(label='Absolute Correlation')
        plt.title('Feature Importance in Encoded Space')
        plt.xlabel('Encoded Dimension')
        plt.ylabel('Feature Index')
        plt.xticks([0, 1], ['Dim 1', 'Dim 2'])
        plt.yticks(range(len(feature_names)), feature_names, rotation=0)
        
        # 7. Reconstruction Error Distribution by Feature
        plt.subplot(337)
        error_distribution = np.square(original_data - reconstructed_data)
        plt.boxplot([error_distribution[:, i] for i in range(min(10, len(feature_names)))],
                   labels=feature_names[:10])
        plt.title('Reconstruction Error Distribution by Feature (Top 10)')
        plt.xlabel('Features')
        plt.ylabel('Squared Error')
        plt.xticks(rotation=45, ha='right')
        
        # 8. Encoded Space with Feature Overlay
        plt.subplot(338)
        scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], 
                            c=original_data[:, feature_idx], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label=feature_names[feature_idx])
        plt.title('Encoded Space with Feature Overlay')
        plt.xlabel('First Encoded Dimension')
        plt.ylabel('Second Encoded Dimension')
        
        # 9. Cumulative Reconstruction Error
        plt.subplot(339)
        sorted_errors = np.sort(np.mean(error_distribution, axis=1))
        plt.plot(range(len(sorted_errors)), sorted_errors)
        plt.title('Cumulative Reconstruction Error Distribution')
        plt.xlabel('Sample Index')
        plt.ylabel('Mean Squared Error')
        
        # Adjust layout and display
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        # Print additional statistics
        print("\nAdvanced Autoencoder Statistics:")
        print(f"Most variable feature: {feature_names[feature_idx]}")
        print(f"Average reconstruction error: {np.mean(feature_errors):.4f}")
        print(f"Max reconstruction error: {np.max(feature_errors):.4f}")
        print(f"Min reconstruction error: {np.min(feature_errors):.4f}")
        
        # Print top 5 most important features for each dimension
        print("\nTop 5 most important features for each encoded dimension:")
        for i in range(2):
            top_features = np.argsort(np.abs(correlations[:, i]))[-5:][::-1]
            print(f"\nDimension {i+1}:")
            for idx in top_features:
                print(f"{feature_names[idx]}: {correlations[idx, i]:.4f}")
                
    except Exception as e:
        print(f"Error in advanced plotting: {e}")
        raise

def analyze_encoding_dimensions(
    data: np.ndarray,
    max_dim: int = 10,
    title: str = "Encoding Dimension Analysis"
) -> None:
    """
    Analyze different encoding dimensions to find the optimal number.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to analyze
    max_dim : int, default=10
        Maximum number of encoding dimensions to test
    title : str, default="Encoding Dimension Analysis"
        Title for the plots
    """
    try:
        # Initialize arrays to store metrics
        reconstruction_errors = []
        explained_variances = []
        information_scores = []
        
        # Test different encoding dimensions
        for dim in range(1, max_dim + 1):
            # Create and train autoencoder
            encoder, encoded_data, reconstructed_data, metrics = autoencoder_clustering(
                data=data,
                encoding_dim=dim
            )
            
            # Calculate reconstruction error
            mse = np.mean(np.square(data - reconstructed_data))
            reconstruction_errors.append(mse)
            
            # Calculate explained variance ratio
            total_var = np.var(data, axis=0).sum()
            encoded_var = np.var(encoded_data, axis=0).sum()
            explained_variances.append(encoded_var / total_var)
            
            # Calculate information preservation score
            # Using correlation between original and reconstructed data
            correlations = []
            for i in range(data.shape[1]):
                corr = np.corrcoef(data[:, i], reconstructed_data[:, i])[0, 1]
                correlations.append(corr)
            information_scores.append(np.mean(correlations))
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Reconstruction Error
        plt.subplot(221)
        plt.plot(range(1, max_dim + 1), reconstruction_errors, 'bo-')
        plt.title('Reconstruction Error vs Encoding Dimensions')
        plt.xlabel('Number of Encoding Dimensions')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        
        # 2. Explained Variance
        plt.subplot(222)
        plt.plot(range(1, max_dim + 1), explained_variances, 'ro-')
        plt.title('Explained Variance Ratio vs Encoding Dimensions')
        plt.xlabel('Number of Encoding Dimensions')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        
        # 3. Information Preservation
        plt.subplot(223)
        plt.plot(range(1, max_dim + 1), information_scores, 'go-')
        plt.title('Information Preservation vs Encoding Dimensions')
        plt.xlabel('Number of Encoding Dimensions')
        plt.ylabel('Average Feature Correlation')
        plt.grid(True)
        
        # 4. Combined Score
        plt.subplot(224)
        # Normalize scores
        norm_errors = (np.max(reconstruction_errors) - reconstruction_errors) / (np.max(reconstruction_errors) - np.min(reconstruction_errors))
        norm_info = (information_scores - np.min(information_scores)) / (np.max(information_scores) - np.min(information_scores))
        combined_score = (norm_errors + norm_info + explained_variances) / 3
        plt.plot(range(1, max_dim + 1), combined_score, 'mo-')
        plt.title('Combined Score vs Encoding Dimensions')
        plt.xlabel('Number of Encoding Dimensions')
        plt.ylabel('Combined Score')
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        # Print analysis results
        print("\nEncoding Dimension Analysis Results:")
        print(f"Best dimension based on reconstruction error: {np.argmin(reconstruction_errors) + 1}")
        print(f"Best dimension based on explained variance: {np.argmax(explained_variances) + 1}")
        print(f"Best dimension based on information preservation: {np.argmax(information_scores) + 1}")
        print(f"Best dimension based on combined score: {np.argmax(combined_score) + 1}")
        
        # Calculate elbow points
        def find_elbow(x, y):
            nPoints = len(x)
            allCoord = np.vstack((x, y)).T
            firstPoint = allCoord[0]
            lineVec = allCoord[-1] - allCoord[0]
            lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
            vecFromFirst = allCoord - firstPoint
            scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
            vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
            vecToLine = vecFromFirst - vecFromFirstParallel
            distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
            idxOfBestPoint = np.argmax(distToLine)
            return idxOfBestPoint + 1
        
        elbow_reconstruction = find_elbow(range(1, max_dim + 1), reconstruction_errors)
        elbow_variance = find_elbow(range(1, max_dim + 1), explained_variances)
        
        print("\nElbow Analysis:")
        print(f"Elbow point for reconstruction error: {elbow_reconstruction} dimensions")
        print(f"Elbow point for explained variance: {elbow_variance} dimensions")
        
        # Recommend optimal dimension
        optimal_dim = max(elbow_reconstruction, elbow_variance)
        print(f"\nRecommended number of encoding dimensions: {optimal_dim}")
        print("This recommendation is based on:")
        print("1. The elbow method for reconstruction error")
        print("2. The elbow method for explained variance")
        print("3. The combined score of all metrics")
        
    except Exception as e:
        print(f"Error in dimension analysis: {e}")
        raise 