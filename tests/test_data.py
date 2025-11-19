import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer

class TestDataPreprocessing:
    def test_missing_data_handling(self):
        """Test missing data handling"""
        config = {'data_preprocessing': {'missing_data_strategy': 'median'}}
        preprocessor = DataPreprocessor(config)
        
        # Test data with missing values
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': ['A', 'B', 'C', np.nan, 'E'],
            'target': [0, 1, 0, 1, 0]
        })
        
        result = preprocessor.handle_missing_data(df)
        assert result.isnull().sum().sum() == 0
    
    def test_data_preparation(self):
        """Test complete data preparation"""
        config = {
            'data_preprocessing': {
                'missing_data_strategy': 'median',
                'imbalance_handling': 'none'
            }
        }
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'num2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'cat1': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        X_train, X_test, y_train, y_test, features = preprocessor.prepare_data(df, 'target')
        
        assert X_train.shape[0] == 8  # 80% of 10
        assert X_test.shape[0] == 2   # 20% of 10
        assert len(features) == 3     # num1, num2, cat1

class TestFeatureEngineering:
    def test_feature_selection(self):
        """Test feature selection methods"""
        config = {'feature_selection': {'method': 'embedded'}}
        feature_engineer = FeatureEngineer(config)
        
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'feature4': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)
        
        selected_features = feature_engineer.select_features_embedded(X, y)
        assert len(selected_features) > 0
        assert all(feat in X.columns for feat in selected_features)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])