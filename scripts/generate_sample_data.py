# scripts/generate_sample_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def generate_sample_data():
    """Generate a sample dataset for the project"""
    # Create directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=42
    )
    
    # Create DataFrame with meaningful column names
    feature_columns = [f'feature_{i:02d}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_columns)
    df['target'] = y
    
    # Introduce some missing values and noise to make it realistic
    np.random.seed(42)
    for col in ['feature_05', 'feature_12', 'feature_18']:
        mask = np.random.random(len(df)) < 0.1  # 10% missing
        df.loc[mask, col] = np.nan
    
    # Add some categorical features
    df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df['sub_category'] = np.random.choice(['X', 'Y'], size=len(df))
    
    # Save to CSV
    df.to_csv('data/raw/sample_data.csv', index=False)
    print(f"Sample dataset generated with {len(df)} rows and {len(df.columns)} columns")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df

if __name__ == "__main__":
    generate_sample_data()