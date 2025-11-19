#!/usr/bin/env python3
"""
Model Drift Monitoring Script
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.deployment.monitor import ModelMonitor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def monitor_drift():
    """Monitor for model and data drift"""
    logger.info("Starting drift monitoring")
    
    try:
        reference_data = pd.read_csv('data/raw/sample_data.csv')
        current_data = reference_data.copy()
        np.random.seed(42)
        drift_features = ['feature_01', 'feature_05', 'feature_10']
        for feature in drift_features:
            if feature in current_data.columns:
                noise = np.random.normal(0, 0.5, len(current_data))
                current_data[feature] = current_data[feature] + noise
        monitor = ModelMonitor(reference_data)
        report = monitor.generate_monitoring_report(current_data)
        
        print("\n" + "="*60)
        print("DRIFT MONITORING REPORT")
        print("="*60)
        if report.get('data_drift', {}).get('overall_drift_detected', False):
            print("DATA DRIFT DETECTED!")
            drifted_features = report['data_drift']['drifted_features']
            print(f"Drifted features: {drifted_features}")
            print(f"Drift ratio: {report['data_drift']['drift_ratio']:.2%}")
        else:
            print("No significant data drift detected")
        
        quality = report.get('data_quality', {})
        print(f"\nData Quality:")
        print(f"  - Missing values: {quality.get('total_missing', 0)}")
        print(f"  - Duplicates: {quality.get('duplicates', 0)}")
        print(f"  - Shape: {quality.get('shape', {})}")
        
        logger.info("Drift monitoring completed")
        
    except Exception as e:
        logger.error(f"Drift monitoring failed: {str(e)}")
        raise

if __name__ == "__main__":
    monitor_drift()