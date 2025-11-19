import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
    logger.info("imblearn is available")
except ImportError as e:
    IMBLEARN_AVAILABLE = False
    logger.warning(f"imblearn not available: {e}")

class DataPreprocessor:
    """Handles missing data, imbalanced data, and feature preprocessing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.imputer = None
        self.scaler = None
        self.encoders = {}
        self.label_encoders = {}
        
    def handle_missing_data(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values using various strategies"""
        df_clean = df.copy()
        
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        if strategy == 'remove':
            df_clean = df_clean.dropna(subset=numerical_cols)
        elif strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
        else:
            self.imputer = SimpleImputer(strategy=strategy)
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])

        df_clean[categorical_cols] = df_clean[categorical_cols].fillna('missing')
        
        logger.info(f"Handled missing data using {strategy} strategy")
        return df_clean
    
    def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series, method: str = 'none') -> Tuple[pd.DataFrame, pd.Series]:
        """Handle imbalanced datasets"""
        if method == 'none':
            return X, y
            
        if not IMBLEARN_AVAILABLE:
            logger.warning("imblearn not available, using basic undersampling")
            from collections import Counter
            counter = Counter(y)
            min_count = min(counter.values())
            
            balanced_indices = []
            for class_label in counter.keys():
                class_indices = np.where(y == class_label)[0]
                selected_indices = np.random.choice(class_indices, min_count, replace=False)
                balanced_indices.extend(selected_indices)
            
            X_resampled = X.iloc[balanced_indices]
            y_resampled = y.iloc[balanced_indices]
            logger.info("Applied basic undersampling for imbalance handling")
            return X_resampled, y_resampled
        
        try:
            if method == 'smote':
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            elif method == 'undersample':
                undersampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(X, y)
            else:
                return X, y
                
            logger.info(f"Applied {method} for imbalance handling")
            return X_resampled, y_resampled
        except Exception as e:
            logger.warning(f"Imblearn method {method} failed: {e}. Using basic undersampling.")
            return self.handle_imbalanced_data(X, y, 'none')
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df_encoded[col].nunique() <= 10: 
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                top_categories = df_encoded[col].value_counts().head(10).index
                df_encoded[col] = df_encoded[col].apply(lambda x: x if x in top_categories else 'other')
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def prepare_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Tuple:
        """Complete data preparation pipeline"""
        logger.info("Starting data preparation pipeline")
        
        strategy = self.config.get('data_preprocessing', {}).get('missing_data_strategy', 'median')
        df_clean = self.handle_missing_data(df, strategy)

        df_encoded = self.encode_categorical_features(df_clean)
        
        if target_col not in df_encoded.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
    
        imbalance_method = self.config.get('data_preprocessing', {}).get('imbalance_handling', 'none')
        if imbalance_method != 'none':
            X, y = self.handle_imbalanced_data(X, y, imbalance_method)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data preparation completed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()