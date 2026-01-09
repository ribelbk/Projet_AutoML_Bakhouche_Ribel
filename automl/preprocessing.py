import numpy as np
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def infer_task_type(y):
    """
    Infer whether the task is regression or classification based on the target variable y.
    
    Parameters:
    -----------
    y : array-like
        Target variable. Can be 1D or 2D numpy array.
    
    Returns:
    --------
    task_type : str
        Either 'classification' or 'regression'
    
    Raises:
    -------
    ValueError: If target array is empty
    """
    # Convert to numpy array if needed
    y = np.asarray(y)
    
    # Handle empty array
    if y.size == 0:
        raise ValueError("Target array is empty")
    
    # Get the shape
    y_shape = y.shape
    
    # Handle 1D arrays (most common case)
    if len(y_shape) == 1:
        # Single dimension array
        return _analyze_1d_array(y)
    
    # Handle 2D arrays
    elif len(y_shape) == 2:
        if y_shape[1] > 1:
            # Multi-column target: likely one-hot encoded classification
            return 'classification'
        else:
            # Single column in 2D array - flatten and analyze
            return _analyze_1d_array(y.ravel())
    
    # Handle 0D or >2D arrays (should be rare)
    else:
        # Flatten everything
        return _analyze_1d_array(y.ravel())


def _analyze_1d_array(y_flat):
    """
    Analyze a 1D array to determine if it's classification or regression.
    
    Parameters:
    -----------
    y_flat : numpy array
        1D array of target values
    
    Returns:
    --------
    str : 'classification' or 'regression'
    """
    # Get unique values
    unique_vals = np.unique(y_flat)
    n_unique = len(unique_vals)
    
    # If array is all NaN, can't determine
    if n_unique == 1 and np.isnan(unique_vals[0]):
        return 'regression'  # Default to regression
    
    # Heuristic 1: Very few unique values -> classification
    if n_unique < 10:
        return 'classification'
    
    # Heuristic 2: Check data type
    if np.issubdtype(y_flat.dtype, np.integer):
        # Integer data
        
        # Heuristic 2a: Integers with moderate number of unique values
        if n_unique < 50:
            return 'classification'
        
        # Heuristic 2b: Check if values look like class labels (0, 1, 2, ...)
        sorted_vals = np.sort(unique_vals)
        is_consecutive = np.all(np.diff(sorted_vals) == 1)
        
        if is_consecutive and sorted_vals[0] >= 0 and sorted_vals[-1] < 100:
            return 'classification'
    
    # Heuristic 3: Check if values are all integers (even if float dtype)
    if np.all(np.equal(np.mod(y_flat, 1), 0)):
        # All values are integers
        if n_unique < 20:
            return 'classification'
    
    # Heuristic 4: Check for binary classification (0/1 or -1/1)
    if n_unique == 2:
        if set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({-1, 1}):
            return 'classification'
    
    # Default to regression
    return 'regression'


def infer_task_type_simple(y):
    """
    Simplified version of task type inference.
    More robust for edge cases.
    
    Parameters:
    -----------
    y : array-like
        Target variable
    
    Returns:
    --------
    str : 'classification' or 'regression'
    """
    try:
        y_array = np.asarray(y)
        
        if y_array.size == 0:
            raise ValueError("Empty target array")
        
        # Handle shape safely
        if hasattr(y_array, 'shape'):
            if len(y_array.shape) == 2 and y_array.shape[1] > 1:
                return 'classification'
        
        # Flatten for analysis
        y_flat = y_array.ravel()
        
        # Count unique values (ignoring NaN)
        valid_mask = ~np.isnan(y_flat)
        if not np.any(valid_mask):
            return 'regression'  # Default if all NaN
        
        y_valid = y_flat[valid_mask]
        n_unique = len(np.unique(y_valid))
        
        # Simple heuristic
        if n_unique <= 20:
            return 'classification'
        else:
            return 'regression'
            
    except Exception as e:
        # If anything fails, default to regression
        print(f"Warning: Could not infer task type: {e}. Defaulting to regression.")
        return 'regression'


def build_pipeline(feature_types, n_features, is_sparse=False):
    """
    Build a scikit-learn preprocessing pipeline.
    
    Parameters:
    -----------
    feature_types : list or None
        List of feature types ('Numerical' or 'Categorical')
    n_features : int
        Number of features in the dataset
    is_sparse : bool
        If True, uses a memory-efficient pipeline for sparse data
    
    Returns:
    --------
    pipeline : sklearn Pipeline or ColumnTransformer
        Preprocessing pipeline
    """
    # 1. Pipeline optimisé pour les données Sparse (Mémoire critique)
    if is_sparse:
        # Pour les données sparse, on ne peut pas centrer la moyenne (with_mean=True)
        # car cela rendrait la matrice dense et exploserait la RAM.
        # MaxAbsScaler est idéal pour les données sparse.
        # Note: Sparse data typically doesn't have missing values in this format
        return Pipeline(steps=[
            ('scaler', MaxAbsScaler())
        ])

    # 2. Pipeline standard pour les données Denses (CSV classiques)
    transformers = []
    
    # Determine feature indices based on types
    if feature_types and len(feature_types) == n_features:
        num_indices = [i for i, t in enumerate(feature_types) 
                      if t and ('Numerical' in t or 'numeric' in t.lower())]
        cat_indices = [i for i, t in enumerate(feature_types) 
                      if t and ('Categorical' in t or 'categorical' in t.lower())]
    else:
        # Fallback: treat all as numerical if types are not provided or don't match
        num_indices = list(range(n_features))
        cat_indices = []
    
    # Preprocessing for numerical data
    if num_indices:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_transformer, num_indices))

    # Preprocessing for categorical data
    if cat_indices:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, cat_indices))

    if transformers:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Keep features that weren't specified
        )
        
        # Wrap in a pipeline for consistency
        return Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
    else:
        # Default pipeline if no transformers were created
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])


def get_preprocessing_summary(feature_types, n_features, is_sparse):
    """
    Get a summary of the preprocessing pipeline that will be used.
    
    Parameters:
    -----------
    feature_types : list or None
        List of feature types
    n_features : int
        Number of features
    is_sparse : bool
        Whether data is sparse
    
    Returns:
    --------
    summary : dict
        Summary of preprocessing configuration
    """
    if is_sparse:
        return {
            'data_type': 'sparse',
            'preprocessing': 'MaxAbsScaler only',
            'reason': 'Sparse data cannot be centered without becoming dense'
        }
    
    if feature_types and len(feature_types) == n_features:
        num_count = sum(1 for t in feature_types 
                       if t and ('Numerical' in t or 'numeric' in t.lower()))
        cat_count = sum(1 for t in feature_types 
                       if t and ('Categorical' in t or 'categorical' in t.lower()))
        
        return {
            'data_type': 'dense',
            'numerical_features': num_count,
            'categorical_features': cat_count,
            'unknown_features': n_features - num_count - cat_count,
            'numerical_preprocessing': 'Imputation (mean) + StandardScaler',
            'categorical_preprocessing': 'Imputation (most frequent) + OneHotEncoder'
        }
    else:
        return {
            'data_type': 'dense',
            'feature_types': 'not provided or mismatched',
            'all_features_treated_as': 'numerical',
            'preprocessing': 'Imputation (mean) + StandardScaler'
        }
