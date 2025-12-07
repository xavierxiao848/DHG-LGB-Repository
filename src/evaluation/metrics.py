"""
Evaluation metrics for metabolite-disease association prediction.

This module implements all metrics reported in the paper, including:
- Accuracy (ACC)
- Sensitivity (SEN) / Recall
- Specificity (SPE)
- Precision (PRE)
- Matthews Correlation Coefficient (MCC) - PRIMARY METRIC
- Area Under ROC Curve (AUC)
- Area Under Precision-Recall Curve (AUPRC)
- F1 Score

References
----------
.. [1] Matthews, B. W. (1975). Comparison of the predicted and observed secondary
       structure of T4 phage lysozyme. Biochimica et Biophysica Acta, 405(2), 442-451.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, Tuple, Optional
import scipy.stats as stats


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (values in [0, 1])
        Required for AUC and AUPRC computation

    Returns
    -------
    dict
        Dictionary containing all metrics:
        - accuracy: Overall classification accuracy
        - sensitivity: True positive rate (recall)
        - specificity: True negative rate
        - precision: Positive predictive value
        - mcc: Matthews Correlation Coefficient (MOST IMPORTANT)
        - f1_score: Harmonic mean of precision and recall
        - auc: Area Under ROC Curve (if y_pred_proba provided)
        - auprc: Area Under Precision-Recall Curve (if y_pred_proba provided)

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    >>> metrics = compute_metrics(y_true, y_pred, y_proba)
    >>> print(f"MCC: {metrics['mcc']:.4f}")

    Notes
    -----
    MCC is considered the most appropriate metric for imbalanced binary classification
    as it takes into account all elements of the confusion matrix and provides a
    balanced measure even when class sizes differ substantially.
    """
    metrics = {}

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)  # TPR / Recall
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)  # PPV
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    # Matthews Correlation Coefficient (MCC) - MOST IMPORTANT
    # MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # Range: [-1, +1] where +1 = perfect prediction, 0 = random, -1 = total disagreement
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics['mcc'] = numerator / denominator if denominator != 0 else 0.0

    # Threshold-independent metrics (require probabilities)
    if y_pred_proba is not None:
        # AUC (Area Under ROC Curve)
        # Measures discriminative ability across all classification thresholds
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc'] = 0.0

        # AUPRC (Area Under Precision-Recall Curve)
        # Emphasizes positive class performance, less sensitive to class imbalance
        try:
            metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auprc'] = 0.0
    else:
        metrics['auc'] = None
        metrics['auprc'] = None

    return metrics


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for a metric across folds.

    Parameters
    ----------
    values : np.ndarray
        Metric values from different folds
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)

    Returns
    -------
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval

    Examples
    --------
    >>> mcc_values = np.array([0.92, 0.93, 0.91, 0.94, 0.92])
    >>> ci_lower, ci_upper = compute_confidence_interval(mcc_values)
    >>> print(f"MCC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    """
    n = len(values)
    mean = np.mean(values)
    std_error = stats.sem(values)  # Standard error of the mean

    # T-distribution critical value
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)

    # Confidence interval
    margin_of_error = t_critical * std_error
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return ci_lower, ci_upper


def print_metrics_summary(
    fold_results: list,
    confidence: float = 0.95
) -> None:
    """
    Print comprehensive summary of cross-validation results.

    Parameters
    ----------
    fold_results : list
        List of metric dictionaries from each fold
    confidence : float, default=0.95
        Confidence level for confidence intervals

    Examples
    --------
    >>> # After cross-validation
    >>> print_metrics_summary(cv_results['fold_results'])
    """
    metric_names = ['accuracy', 'sensitivity', 'specificity', 'precision', 'mcc', 'auc', 'auprc']

    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Metric':<20} {'Mean':<12} {'Std':<12} {f'{int(confidence*100)}% CI':<20}")
    print("-"*70)

    for metric in metric_names:
        values = np.array([fold[metric] for fold in fold_results if fold[metric] is not None])

        if len(values) == 0:
            continue

        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower, ci_upper = compute_confidence_interval(values, confidence)

        # Highlight MCC as primary metric
        prefix = ">>> " if metric == 'mcc' else "    "

        print(f"{prefix}{metric.upper():<17} {mean_val:>6.4f}      ±{std_val:<6.4f}    "
              f"[{ci_lower:.4f}, {ci_upper:.4f}]")

    print("="*70)
    print("\nNOTE: MCC (Matthews Correlation Coefficient) is the primary metric")
    print("      as it provides the most reliable assessment for imbalanced datasets.")
    print("="*70 + "\n")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve with AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    title : str, default="ROC Curve"
        Plot title
    save_path : str, optional
        Path to save the plot (if None, display only)
    """
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    else:
        plt.show()


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot Precision-Recall curve with AUPRC score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    title : str, default="Precision-Recall Curve"
        Plot title
    save_path : str, optional
        Path to save the plot (if None, display only)
    """
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    auprc_score = average_precision_score(y_true, y_pred_proba)

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'AUPRC = {auprc_score:.4f}')
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                label=f'Random (AUPRC = {baseline:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    else:
        plt.show()


def statistical_comparison(
    values_a: np.ndarray,
    values_b: np.ndarray,
    test: str = 't-test',
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform statistical significance test between two sets of metric values.

    Parameters
    ----------
    values_a : np.ndarray
        Metric values from method A (e.g., DHG-LGB)
    values_b : np.ndarray
        Metric values from method B (e.g., XGBoost)
    test : str, default='t-test'
        Statistical test: 't-test' or 'wilcoxon'
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary containing:
        - statistic: Test statistic
        - p_value: P-value
        - significant: Whether difference is significant (p < alpha)

    Examples
    --------
    >>> lgb_mcc = np.array([0.930, 0.931, 0.929, 0.932, 0.930])
    >>> xgb_mcc = np.array([0.927, 0.928, 0.926, 0.929, 0.927])
    >>> result = statistical_comparison(lgb_mcc, xgb_mcc)
    >>> print(f"P-value: {result['p_value']:.4f}, Significant: {result['significant']}")
    """
    if test == 't-test':
        statistic, p_value = stats.ttest_rel(values_a, values_b)
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(values_a, values_b)
    else:
        raise ValueError(f"Unknown test: {test}. Use 't-test' or 'wilcoxon'.")

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha
    }
