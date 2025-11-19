import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature selection and engineering techniques"""
    
    def __init__(self, config: dict):
        self.config = config
        self.selected_features = []
        self.selector = None
        
    def create_interaction_features(self, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df_engineered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols[:top_k]):
                for col2 in numerical_cols[i+1:top_k]:
                    if col1 != col2:
                        df_engineered[f'{col1}_x_{col2}'] = df_engineered[col1] * df_engineered[col2]
                        df_engineered[f'{col1}_div_{col2}'] = df_engineered[col1] / (df_engineered[col2] + 1e-8)  
        
        logger.info(f"Created interaction features")
        return df_engineered
    
    def select_features_filter(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """Feature selection using filter methods"""
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.selector = selector
        
        logger.info(f"Selected {len(self.selected_features)} features using filter method")
        return X_selected
    
    def select_features_wrapper(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """Feature selection using wrapper methods (RFE)"""
        k = min(k, X.shape[1])
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=k)
        X_selected = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.selector = selector
        
        logger.info(f"Selected {len(self.selected_features)} features using wrapper method")
        return X_selected
    
    def select_features_embedded(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.01) -> List[str]:
        """Feature selection using embedded methods (feature importance)"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.selected_features = feature_importance[
            feature_importance['importance'] > threshold
        ]['feature'].tolist()

        if not self.selected_features:
            self.selected_features = feature_importance.head(5)['feature'].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features using embedded method")
        return self.selected_features
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get feature importance from multiple methods"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'random_forest_importance': rf_importance,
            'mutual_information': mi_scores
        })
        
        importance_df['combined_score'] = (
            importance_df['random_forest_importance'] + importance_df['mutual_information']
        )
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        
        return importance_df