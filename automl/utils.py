"""
Utility functions for AutoML.
"""
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import warnings
from tqdm import tqdm
import contextlib
import joblib


def load_data(filepath, debug=False):
    """
    Load data from various file formats.
    
    Supports:
    1. Sparse format (index:value)
    2. CSV/TSV format
    3. Space-separated values
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    debug : bool
        Whether to print debug information
    
    Returns:
    --------
    data : numpy array or scipy sparse matrix
        Loaded data
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.endswith('.data'):
            return _load_sparse_or_dense(filepath, debug)
        elif filepath.endswith('.solution'):
            return _load_solution_file(filepath, debug)
        else:
            # Default to pandas for other files
            return pd.read_csv(filepath, sep=r'\s+', header=None).values
            
    except Exception as e:
        if debug:
            print(f"Error loading {filepath}: {e}")
        return np.array([])


def _load_sparse_or_dense(filepath, debug=False):
    """
    Load data file that can be in sparse or dense format.
    """
    try:
        # Try sparse format first
        data = []
        indices = []
        indptr = [0]
        n_features = 0
        n_lines = 0
        sparse_format_detected = False
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                items = line.split()
                
                # Check if this line has sparse format
                has_sparse = any(':' in item for item in items)
                
                if has_sparse:
                    sparse_format_detected = True
                    for item in items:
                        if ':' in item:
                            idx_str, val_str = item.split(':', 1)
                            idx = int(idx_str) - 1  # Convert to 0-based
                            val = float(val_str)
                            indices.append(idx)
                            data.append(val)
                            if idx + 1 > n_features:
                                n_features = idx + 1
                else:
                    # Dense format
                    for col_idx, val_str in enumerate(items):
                        val = float(val_str)
                        indices.append(col_idx)
                        data.append(val)
                        if col_idx + 1 > n_features:
                            n_features = col_idx + 1
                
                indptr.append(len(indices))
                n_lines += 1
        
        if sparse_format_detected and len(data) > 0:
            X = csr_matrix((data, indices, indptr), shape=(n_lines, n_features))
            if debug:
                print(f"  Loaded sparse matrix: {X.shape}, density: {X.nnz/(X.shape[0]*X.shape[1]):.3f}")
            return X
        else:
            # Fallback to dense loading
            return pd.read_csv(filepath, sep=r'\s+', header=None).values
            
    except Exception as e:
        if debug:
            print(f"  Error in sparse loading: {e}")
        # Fallback to pandas
        return pd.read_csv(filepath, sep=r'\s+', header=None).values


def _load_solution_file(filepath, debug=False):
    """
    Load solution/target file.
    """
    try:
        # Try reading with pandas first
        data = pd.read_csv(filepath, sep=r'\s+', header=None).values
        
        # Handle special cases
        if data.shape[1] == 1:
            # Single column - flatten
            data = data.ravel()
        elif data.shape[1] > 1 and np.all(np.sum(data, axis=1) == 1):
            # One-hot encoded - convert to labels
            data = np.argmax(data, axis=1)
        
        return data
        
    except Exception as e:
        if debug:
            print(f"  Error loading solution file: {e}")
        
        # Manual loading as fallback
        try:
            with open(filepath, 'r') as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if line:
                        values = [float(v) for v in line.split()]
                        lines.append(values)
                
                if not lines:
                    return np.array([])
                
                data = np.array(lines)
                if data.shape[1] == 1:
                    data = data.ravel()
                
                return data
        except:
            return np.array([])


def load_types(filepath):
    """
    Load feature types from .type file.
    
    Parameters:
    -----------
    filepath : str
        Path to .type file
    
    Returns:
    --------
    types : list or None
        List of feature types, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            types = [line.strip() for line in f if line.strip()]
        return types
    except:
        return None


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def setup_debug_mode(debug):
    """
    Setup warning filters based on debug mode.
    
    Parameters:
    -----------
    debug : bool
        Whether debug mode is enabled
    """
    if debug:
        warnings.filterwarnings('default')
        print("DEBUG MODE ENABLED")
    else:
        warnings.filterwarnings('ignore')


def check_required_files(dataset_path, dataset_name, debug=False):
    """
    Check if required dataset files exist.
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    dataset_name : str
        Name of the dataset
    debug : bool
        Whether to print debug information
    
    Returns:
    --------
    bool
        True if all required files exist, False otherwise
    """
    required_files = [
        f"{dataset_name}.data",
        f"{dataset_name}.solution"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(dataset_path, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files and debug:
        print(f"  Missing files: {', '.join(missing_files)}")
    
    return len(missing_files) == 0
