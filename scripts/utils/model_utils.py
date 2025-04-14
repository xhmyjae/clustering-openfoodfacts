import joblib
import os
from typing import Optional
from sklearn.base import BaseEstimator
from datetime import datetime

def save_model(
    model: BaseEstimator,
    model_name: str,
    model_dir: str = 'models'
) -> str:
    """
    Save a trained model to disk with timestamp in the filename.
    
    Parameters:
    -----------
    model : BaseEstimator
        The trained model to save
    model_name : str
        Name of the model (without extension)
    model_dir : str, default='models'
        Directory to save the model in
    
    Returns:
    --------
    str
        Path to the saved model file
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Get current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the full path for the model file with timestamp
    model_path = os.path.join(model_dir, f'{model_name}_{timestamp}.joblib')
    
    # Save the model
    joblib.dump(model, model_path)
    
    print(f"Model saved to: {model_path}")
    return model_path

def load_model(
    model_name: str,
    model_dir: str = 'models'
) -> Optional[BaseEstimator]:
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (without extension)
    model_dir : str, default='models'
        Directory where the model is saved
    
    Returns:
    --------
    BaseEstimator or None
        The loaded model, or None if the model doesn't exist
    """
    # Look for the most recent model file matching the name pattern
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name) and f.endswith('.joblib')]
    
    if not model_files:
        print(f"No model files found for: {model_name}")
        return None
    
    # Sort by name (which includes timestamp) and get the most recent
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None 