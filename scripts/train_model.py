#!/usr/bin/env python3
"""
Complete ML Pipeline Training Script
"""
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.utils.config import config
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

def main():
    """Main training pipeline"""
    logger.info("Starting ML Pipeline Training")
    
    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        data_path = 'data/raw/sample_data.csv'
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            logger.info("Generating sample data...")
            from scripts.generate_sample_data import generate_sample_data
            generate_sample_data()
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Step 2: Data preprocessing
        logger.info("Step 2: Data preprocessing")
        preprocessor = DataPreprocessor(config.get_model_config())
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df, 'target')
        
        # Convert back to DataFrame for feature engineering
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Step 3: Feature engineering
        logger.info("Step 3: Feature engineering")
        feature_engineer = FeatureEngineer(config.get_model_config())
        
        # Create interaction features
        X_train_engineered = feature_engineer.create_interaction_features(X_train_df)
        X_test_engineered = feature_engineer.create_interaction_features(X_test_df)
        
        # Feature selection
        selected_features = feature_engineer.select_features_embedded(X_train_engineered, y_train)
        X_train_selected = X_train_engineered[selected_features]
        X_test_selected = X_test_engineered[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Step 4: Model training
        logger.info("Step 4: Model training")
        trainer = ModelTrainer(config.get_model_config())
        results = trainer.train_models(X_train_selected, y_train, X_test_selected, y_test)
        
        # Step 5: Model evaluation
        logger.info("Step 5: Model evaluation")
        evaluator = ModelEvaluator()
        report = evaluator.generate_report(trainer.models, X_test_selected, y_test)
        
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best Model: {trainer.best_model_name}")
        print(f"Best CV F1 Score: {results[trainer.best_model_name]['cv_mean']:.4f}")
        print("\nModel Comparison:")
        print(report.to_string(index=False))
        
        logger.info("ML Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()