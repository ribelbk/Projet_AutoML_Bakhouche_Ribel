"""
AutoML package for automatic machine learning.
"""

from .core import fit, eval, AutoMLPredictor
from .models import (
    _get_classification_models, 
    _get_regression_models,
    get_scoring_metric,
    get_cv_folds
)
from .metrics import (
    classification_metrics,
    regression_metrics,
    print_classification_summary,
    print_regression_summary
)
from .utils import (
    load_data,
    load_types,
    setup_debug_mode,
    check_required_files
)

__version__ = "1.0.0"
__author__ = "AutoML Team"
__all__ = [
    'fit',
    'eval',
    'AutoMLPredictor',
    'classification_metrics',
    'regression_metrics',
    'load_data',
    'load_types'
]
