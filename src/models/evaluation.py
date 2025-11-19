# src/models/evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model and return metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Evaluation completed for {model_name}")
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot confusion matrix without seaborn"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Create basic heatmap without seaborn
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'logs/confusion_matrix_{model_name}.png')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name: str):
        """Plot ROC curve"""
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f'logs/roc_curve_{model_name}.png')
            plt.close()
    
    def generate_report(self, models: Dict, X_test, y_test) -> pd.DataFrame:
        """Generate comprehensive evaluation report"""
        report_data = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Generate plots
            self.plot_confusion_matrix(y_test, self.results[model_name]['predictions'], model_name)
            if self.results[model_name]['probabilities'] is not None:
                self.plot_roc_curve(y_test, self.results[model_name]['probabilities'], model_name)
            
            report_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 'N/A')}"
            })
        
        report_df = pd.DataFrame(report_data)
        logger.info("Comprehensive evaluation report generated")
        return report_df