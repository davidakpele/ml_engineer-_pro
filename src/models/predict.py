# src/models/predict.py

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Model prediction interface"""
    
    def __init__(self, model_path: str = "models"):
        self.model_path = model_path
        self.models = {}
        self.expected_features = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models and detect feature count"""
        if os.path.exists(self.model_path):
            for file in os.listdir(self.model_path):
                if file.endswith('.pkl') and file != 'best_model_info.json':
                    model_name = file.replace('.pkl', '')
                    with open(os.path.join(self.model_path, file), 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            logger.info(f"Loaded {len(self.models)} models")
            self._detect_feature_count()
            
        else:
            logger.warning("Models directory not found")
    
    def _detect_feature_count(self):
        """Detect how many features the models actually expect"""
        if not self.models:
            return

        first_model = list(self.models.values())[0]
        
        if hasattr(first_model, 'n_features_in_'):
            self.expected_features = first_model.n_features_in_
            logger.info(f"Detected models expect {self.expected_features} features")
        elif hasattr(first_model, 'n_features_'):
            self.expected_features = first_model.n_features_
            logger.info(f"Detected models expect {self.expected_features} features")
        else:
            self.expected_features = 5 
            logger.warning(f"Could not detect feature count, using default: {self.expected_features}")
    
    def predict(self, features: List[float], model_name: str = "best") -> Dict[str, Any]:
        """Make prediction using specified model"""
        if not self.models:
            raise ValueError("No models loaded")
        
        if self.expected_features is None:
            raise ValueError("Feature count not detected from models")
        
        if len(features) != self.expected_features:
            raise ValueError(f"Feature shape mismatch. Expected: {self.expected_features}, Got: {len(features)}. Please provide exactly {self.expected_features} features.")
        
        if model_name == "best":
            try:
                import json
                with open('models/best_model_info.json', 'r') as f:
                    best_info = json.load(f)
                model_name = best_info['model_name']
            except:
                model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        features_array = np.array(features).reshape(1, -1)
        
        prediction = int(model.predict(features_array)[0])
        probability = float(model.predict_proba(features_array)[0][1]) if hasattr(model, 'predict_proba') else None
        
        return {
            'prediction': prediction,
            'probability': probability,
            'model_used': model_name,
            'features_used': len(features),
            'status': 'success'
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_expected_features(self) -> int:
        """Get the number of features expected by the models"""
        if self.expected_features is None:
            raise ValueError("Feature count not detected")
        return self.expected_features