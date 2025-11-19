import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, reference_data: pd.DataFrame = None):
        self.reference_data = reference_data
        self.drift_threshold = 0.05
        self.performance_history = []
        self.drift_history = []
    
    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for drift detection"""
        self.reference_data = data
        logger.info(f"Reference data set with shape: {data.shape}")
    
    def calculate_psi(self, current_data: pd.DataFrame, feature: str) -> float:
        """Calculate Population Stability Index for a feature"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        ref_series = self.reference_data[feature].dropna()
        curr_series = current_data[feature].dropna()
        
        if len(ref_series) == 0 or len(curr_series) == 0:
            return 0.0
        
        bins = np.histogram_bin_edges(ref_series, bins=10)
        
        ref_dist, _ = np.histogram(ref_series, bins=bins)
        curr_dist, _ = np.histogram(curr_series, bins=bins)
        
        ref_dist = ref_dist / len(ref_series)
        curr_dist = curr_dist / len(curr_series)
 
        psi = 0
        for i in range(len(ref_dist)):
            if ref_dist[i] == 0:
                ref_dist[i] = 0.0001  
            if curr_dist[i] == 0:
                curr_dist[i] = 0.0001
                
            psi += (curr_dist[i] - ref_dist[i]) * np.log(curr_dist[i] / ref_dist[i])
        
        return float(psi)  
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift using multiple statistical tests"""
        if self.reference_data is None:
            raise ValueError("Reference data not set for drift detection")
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'overall_drift_detected': False
        }
        
        numerical_features = current_data.select_dtypes(include=[np.number]).columns
        
        drifted_features = []
        for feature in numerical_features:
            if feature in self.reference_data.columns:
                ks_stat, ks_pvalue = ks_2samp(
                    self.reference_data[feature].dropna(),
                    current_data[feature].dropna()
                )

                psi = self.calculate_psi(current_data, feature)
                
                feature_drifted = psi > self.drift_threshold or ks_pvalue < 0.05
                
                drift_report['features'][feature] = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'psi': float(psi),
                    'drift_detected': bool(feature_drifted)  
                }
                
                if feature_drifted:
                    drifted_features.append(feature)
        
        drift_report['overall_drift_detected'] = bool(len(drifted_features) > 0)  
        drift_report['drifted_features'] = drifted_features
        drift_report['drift_ratio'] = float(len(drifted_features) / len(numerical_features) if len(numerical_features) > 0 else 0.0)
        
        self.drift_history.append(drift_report)
        
        return drift_report
    
    def monitor_performance(self, y_true: List, y_pred: List, model_version: str) -> Dict[str, Any]:
        """Monitor model performance over time"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'sample_size': int(len(y_true))
        }
        
        if len(self.performance_history) > 0:
            current_f1 = metrics['f1_score']
            previous_f1 = self.performance_history[-1]['f1_score']
            metrics['f1_degradation'] = bool(current_f1 < previous_f1 * 0.95) 
            metrics['f1_change'] = float(current_f1 - previous_f1)
        else:
            metrics['f1_degradation'] = False
            metrics['f1_change'] = 0.0
        
        self.performance_history.append(metrics)

        self._save_performance_history()
        
        return metrics
    
    def generate_monitoring_report(self, current_data: pd.DataFrame = None,
                                 y_true: Optional[List] = None, 
                                 y_pred: Optional[List] = None,
                                 model_version: str = "latest") -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
        }
        
        if current_data is not None and self.reference_data is not None:
            report['data_drift'] = self.detect_data_drift(current_data)
        
        report['data_quality'] = self.check_data_quality(current_data) if current_data is not None else {}
        
        if y_true is not None and y_pred is not None:
            report['performance'] = self.monitor_performance(y_true, y_pred, model_version)

        self._save_monitoring_report(report)
        
        logger.info(f"Monitoring report generated for {model_version}")
        return report
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues"""
        if data is None:
            return {}
        
        missing_values = {str(k): int(v) for k, v in data.isnull().sum().to_dict().items()}
        data_types = {str(k): str(v) for k, v in data.dtypes.astype(str).to_dict().items()}
        outliers = {str(k): int(v) for k, v in self.detect_outliers(data).items()}
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'missing_values': missing_values,
            'total_missing': int(data.isnull().sum().sum()),
            'duplicates': int(data.duplicated().sum()),
            'data_types': data_types,
            'outliers': outliers,
            'shape': {'rows': int(len(data)), 'columns': int(len(data.columns))}
        }
        
        return quality_report
    
    def detect_outliers(self, data: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method"""
        outliers = {}
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outliers[str(col)] = int(outlier_count) 
        
        return outliers
    
    def _save_performance_history(self):
        """Save performance history to file"""
        os.makedirs('logs', exist_ok=True)
        with open('logs/performance_history.json', 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
    
    def _save_monitoring_report(self, report: Dict[str, Any]):
        """Save monitoring report to file"""
        os.makedirs('logs', exist_ok=True)
        with open('logs/monitoring_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """Get performance trend over time"""
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        return {
            'performance_trend': df.to_dict('records'),
            'latest_f1': float(df['f1_score'].iloc[-1]) if len(df) > 0 else 0.0,
            'average_f1': float(df['f1_score'].mean()) if len(df) > 0 else 0.0
        }