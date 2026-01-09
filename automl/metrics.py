"""
Evaluation metrics for AutoML.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)


def classification_metrics(y_true, y_pred, y_pred_proba=None, average='weighted'):
    """
    Compute comprehensive classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for ROC-AUC
    average : str
        Average method for multi-class metrics
    
    Returns:
    --------
    metrics : dict
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Class distribution
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    metrics['n_classes'] = len(unique_classes)
    metrics['class_distribution'] = dict(zip(unique_classes, class_counts))
    
    # ROC-AUC if probabilities are provided (binary classification only)
    if y_pred_proba is not None and len(unique_classes) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['roc_auc'] = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    if len(unique_classes) <= 10:  # Only compute if not too many classes
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['detailed_report'] = report
    
    return metrics


def regression_metrics(y_true, y_pred):
    """
    Compute comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    metrics : dict
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Error metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Percentage errors (if all values are positive)
    if np.all(y_true > 0) and np.all(y_pred > 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['mape'] = mape
    
    # R-squared and explained variance
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    
    # Worst-case error
    metrics['max_error'] = max_error(y_true, y_pred)
    
    # Target statistics
    metrics['y_true_mean'] = np.mean(y_true)
    metrics['y_true_std'] = np.std(y_true)
    metrics['y_pred_mean'] = np.mean(y_pred)
    metrics['y_pred_std'] = np.std(y_pred)
    
    # Residual statistics
    residuals = y_true - y_pred
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)
    
    return metrics


def print_classification_summary(metrics, model_name=None):
    """
    Print formatted classification metrics summary.
    
    Parameters:
    -----------
    metrics : dict
        Classification metrics dictionary
    model_name : str, optional
        Name of the model
    """
    if model_name:
        print(f"\n{'='*50}")
        print(f"CLASSIFICATION RESULTS - {model_name}")
        print(f"{'='*50}")
    
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision']:.4f}")
    print(f"Recall (weighted):    {metrics['recall']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 Score (macro):    {metrics['f1_macro']:.4f}")
    
    if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
        print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    
    print(f"\nNumber of classes:  {metrics['n_classes']}")
    print("Class distribution:")
    for cls, count in metrics['class_distribution'].items():
        print(f"  Class {cls}: {count} samples")


def print_regression_summary(metrics, model_name=None):
    """
    Print formatted regression metrics summary.
    
    Parameters:
    -----------
    metrics : dict
        Regression metrics dictionary
    model_name : str, optional
        Name of the model
    """
    if model_name:
        print(f"\n{'='*50}")
        print(f"REGRESSION RESULTS - {model_name}")
        print(f"{'='*50}")
    
    print(f"RMSE:               {metrics['rmse']:.4f}")
    print(f"MAE:                {metrics['mae']:.4f}")
    print(f"R² Score:           {metrics['r2']:.4f}")
    
    if 'mape' in metrics:
        print(f"MAPE:               {metrics['mape']:.2f}%")
    
    print(f"Explained Variance: {metrics['explained_variance']:.4f}")
    print(f"Max Error:          {metrics['max_error']:.4f}")
    
    print(f"\nTarget Statistics:")
    print(f"  True mean: ± {metrics['y_true_mean']:.4f} ({metrics['y_true_std']:.4f})")
    print(f"  Pred mean: ± {metrics['y_pred_mean']:.4f} ({metrics['y_pred_std']:.4f})")
    print(f"  Residuals: ± {metrics['residual_mean']:.4f} ({metrics['residual_std']:.4f})")
