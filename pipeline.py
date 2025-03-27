# Imports
import mlflow
import pandas as pd
import numpy as np


mlflow.autolog()

# Load Open Food Facts dataset csv
path = "data/en.openfoodfacts.org.products.csv"

df = pd.read_csv(path, 
                 sep='\\t', 
                 encoding="utf-8",
                 on_bad_lines='skip',
                 nrows=300000)

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Handle missing values


