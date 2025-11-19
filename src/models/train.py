# src/models/train.py

import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training with comprehensive overfitting prevention"""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.mlflow_experiment = "ml_engineer_portfolio"
        
    def prevent_overfitting_setup(self):
        """Return configurations to prevent overfitting"""
        return {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'min_samples_split': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
        }
    
    def train_models(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train multiple models with proper validation"""
        os.makedirs('models', exist_ok=True)
        mlflow.set_experiment(self.mlflow_experiment)
        
        overfitting_config = self.prevent_overfitting_setup()
        results = {}
        
        models = {
            'random_forest': RandomForestClassifier(**overfitting_config['random_forest']),
            'gradient_boosting': GradientBoostingClassifier(**overfitting_config['gradient_boosting']),
            'xgboost': xgb.XGBClassifier(**overfitting_config['xgboost']),
            'lightgbm': lgb.LGBMClassifier(**overfitting_config['lightgbm'])
        }
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(X_train, y_train)
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
  
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, model_name)
                
                model_path = f'models/{model_name}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                self.models[model_name] = model
                results[model_name] = metrics
                
                logger.info(f"Trained {model_name} - CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = self.models[self.best_model_name]
        
        best_model_info = {
            'model_name': self.best_model_name,
            'metrics': results[self.best_model_name],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open('models/best_model_info.json', 'w') as f:
            import json
            json.dump(best_model_info, f, indent=2)
        
        logger.info(f"Best model: {self.best_model_name} with CV F1: {results[self.best_model_name]['cv_mean']:.4f}")
        return results
    
    def get_best_model(self):
        """Get the best performing model"""
        return self.best_model, self.best_model_name