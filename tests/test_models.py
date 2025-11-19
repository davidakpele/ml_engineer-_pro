import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.train import ModelTrainer
from src.models.evaluation import ModelEvaluator

class TestModelTraining:
    def test_model_training(self):
        """Test model training pipeline"""
        config = {
            'model': {
                'random_forest': {'n_estimators': 10, 'max_depth': 5},
                'gradient_boosting': {'n_estimators': 10, 'max_depth': 3}
            }
        }
        
        trainer = ModelTrainer(config)
        
        # Generate sample data
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 20)
        
        results = trainer.train_models(X_train, y_train, X_test, y_test)
        
        assert 'random_forest' in results
        assert 'gradient_boosting' in results
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None

class TestModelEvaluation:
    def test_model_evaluation(self):
        """Test model evaluation"""
        evaluator = ModelEvaluator()
        
        # Generate sample data and predictions
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        # Mock model
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
            def predict_proba(self, X):
                return np.random.rand(len(X), 2)
        
        model = MockModel()
        metrics = evaluator.evaluate_model(model, X_test, y_test, 'test_model')
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v"])