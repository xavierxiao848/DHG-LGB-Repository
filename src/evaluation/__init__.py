"""Evaluation modules for DHG-LGB framework."""

from .metrics import (
    compute_metrics,
    compute_confidence_interval,
    print_metrics_summary,
    plot_roc_curve,
    plot_pr_curve,
    statistical_comparison,
)

__all__ = [
    'compute_metrics',
    'compute_confidence_interval',
    'print_metrics_summary',
    'plot_roc_curve',
    'plot_pr_curve',
    'statistical_comparison',
]
