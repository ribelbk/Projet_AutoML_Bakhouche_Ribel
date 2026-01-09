import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import contextlib
import joblib
from .preprocessing import build_pipeline, infer_task_type
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from scipy.sparse import csr_matrix, issparse

import warnings
from tqdm import tqdm
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global variable to store trained predictors
trained_predictors = {}

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar class"""
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

def fit(data_dest, dataset_name=None, debug=False):
    """
    Train models on all datasets found in data_dest.
    """
    global trained_predictors
    trained_predictors = {}
    
    if debug:
        warnings.filterwarnings('default')
        print(f"DEBUG MODE ON")
    else:
        warnings.filterwarnings('ignore')
    
    if not os.path.exists(data_dest):
        print(f"Directory {data_dest} does not exist.")
        return

    # Look for subdirectories
    if dataset_name:
        target_path = os.path.join(data_dest, dataset_name)
        if os.path.isdir(target_path):
            subdirs = [dataset_name]
        else:
            print(f"Dataset directory {dataset_name} not found in {data_dest}")
            return
    else:
        subdirs = [d for d in os.listdir(data_dest) if os.path.isdir(os.path.join(data_dest, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {data_dest}.")
        return

    if debug:
        print(f"Found {len(subdirs)} datasets: {', '.join(subdirs)}")

    iterator = subdirs
    if not debug:
        iterator = tqdm(subdirs, desc="Total Progress", unit="dataset")

    for subdir in iterator:
        dataset_path = os.path.join(data_dest, subdir)
        dataset_name = subdir
        
        # Check for required files
        data_file = os.path.join(dataset_path, f"{dataset_name}.data")
        solution_file = os.path.join(dataset_path, f"{dataset_name}.solution")
        
        if os.path.exists(data_file) and os.path.exists(solution_file):
            if debug:
                print(f"\nProcessing dataset: {dataset_name}")

            try:
                predictor = AutoMLPredictor(dataset_name, dataset_path, debug=debug)
                predictor.fit()
                trained_predictors[dataset_name] = predictor
            except Exception as e:
                if debug:
                    import traceback
                    traceback.print_exc()
                    print(f"Error processing {dataset_name}: {e}")
                else:
                     print(f"Error processing {dataset_name}: {e}")
        else:
            if debug:
                print(f"Skipping {dataset_name}: missing .data or .solution file")

def eval():
    """
    Evaluate trained models and print summary.
    """
    global trained_predictors
    if not trained_predictors:
        print("No models trained. Run automl.fit() first.")
        return
    
    print("\n" + "="*40)
    print("       PERFORMANCE SUMMARY       ")
    print("="*40)
    
    for name, predictor in trained_predictors.items():
        print(f"\nDataset: {name}")
        try:
            predictor.evaluate()
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

class AutoMLPredictor:
    def __init__(self, name, path, debug=False):
        self.name = name
        self.path = path
        self.debug = debug
        self.model = None
        self.best_params = None
        self.task_type = None 
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        
    def fit(self):
        try:
            # Load data
            X = self._load_data(os.path.join(self.path, f"{self.name}.data"))
            y = self._load_data(os.path.join(self.path, f"{self.name}.solution"))
            
            if X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError(f"Empty data or solution file for {self.name}")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Mismatched samples: X has {X.shape[0]}, y has {y.shape[0]}")

            # Infer task type
            self.task_type = infer_task_type(y)
            if self.debug:
                print(f"  Task type: {self.task_type}")
                print(f"  Data shape: {X.shape}, Solution shape: {y.shape}")
                print(f"  Is Sparse: {issparse(X)}")
                print(f"  Target unique values: {np.unique(y)}")
            
            # Load feature types (optional)
            type_file = os.path.join(self.path, f"{self.name}.type")
            feature_types = self._load_types(type_file)
            
            # Handle One-Hot Encoded Targets for Classification
            if self.task_type == 'classification' and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            elif y.shape[1] == 1:
                y = y.ravel()
            
            # Check for valid target after processing
            if len(y) == 0:
                raise ValueError(f"No valid target values after processing")
            
            # Ensure target is integer for classification
            if self.task_type == 'classification':
                y = y.astype(int)
            
            if self.debug:
                print(f"  Processed target shape: {y.shape}")
                print(f"  Target range: {y.min()} to {y.max()}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if self.task_type == 'classification' else None
            )
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
            if self.debug:
                print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            # Build pipeline
            is_sparse_data = issparse(X_train)
            pipeline = build_pipeline(feature_types, X_train.shape[1], is_sparse=is_sparse_data)
            
            # Handle small datasets - reduce CV folds
            n_samples = X_train.shape[0]
            cv_folds = min(3, n_samples // 10) if n_samples < 30 else 3
            if cv_folds < 2:
                cv_folds = 2
                
            if self.debug:
                print(f"  Using {cv_folds} CV folds")
            
            # Select models based on task type
            if self.task_type == 'classification':
                models, param_grids = self._get_classification_models(is_sparse_data)
                scoring = 'accuracy'
            else:
                models, param_grids = self._get_regression_models()
                scoring = 'neg_mean_squared_error'
            
            if self.debug:
                print(f"  Available models: {list(models.keys())}")
            
            best_score = -np.inf
            best_model_name = None
            best_model = None
            best_params = None
            
            model_iter = models.items()
            if not self.debug:
                model_iter = tqdm(models.items(), desc=f"Training models ({self.name})", leave=False, unit="model")
            
            for model_name, model in model_iter:
                try:
                    if self.debug:
                        print(f"    Trying {model_name}...")
                    
                    full_pipeline = Pipeline(steps=[('preprocessor', pipeline), ('model', model)])
                    
                    # Use simple validation for very small datasets
                    if n_samples < 50:
                        # Simple train-test split instead of CV
                        search = full_pipeline
                        search.fit(X_train, y_train)
                        # Evaluate on validation set
                        if self.task_type == 'classification':
                            score = accuracy_score(y_train, search.predict(X_train))
                        else:
                            score = -mean_squared_error(y_train, search.predict(X_train))
                        params = {}
                    else:
                        # Use GridSearchCV with appropriate CV
                        search = GridSearchCV(
                            full_pipeline, 
                            param_grids[model_name], 
                            cv=cv_folds, 
                            scoring=scoring, 
                            n_jobs=-1 if n_samples > 100 else 1,
                            verbose=0,
                            error_score='raise'
                        )
                        
                        search.fit(X_train, y_train)
                        score = search.best_score_
                        params = search.best_params_
                    
                    if self.debug:
                        print(f"      {model_name} score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model = search.best_estimator_ if hasattr(search, 'best_estimator_') else search
                        best_params = params
                        
                except Exception as e:
                    if self.debug:
                        print(f"    Error with {model_name}: {str(e)}")
            
            if best_model is None:
                raise ValueError("No model could be trained successfully")
            
            self.model = best_model
            self.best_params = best_params
            
            if self.debug:
                print(f"  Best model: {best_model_name} with score: {best_score:.4f}")
                if best_params:
                    print(f"  Best params: {best_params}")
        
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            raise Exception(f"Error in fit() for {self.name}: {str(e)}")
    
    def _get_classification_models(self, is_sparse):
        """Get classification models and parameter grids"""
        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_jobs=-1),
        }
        
        param_grids = {
            'RandomForest': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10, None],
            },
            'LogisticRegression': {
                'model__C': [0.1, 1, 10],
                'model__penalty': ['l2']
            },
            'KNN': {
                'model__n_neighbors': [3, 5, 7],
            }
        }
        
        # Add linear model for sparse data
        if is_sparse:
            models['LinearSVC'] = LinearSVC(random_state=42, dual=False, max_iter=1000)
            param_grids['LinearSVC'] = {
                'model__C': [0.1, 1, 10],
                'model__penalty': ['l2']
            }
        else:
            models['GaussianNB'] = GaussianNB()
            param_grids['GaussianNB'] = {}
        
        # Add ExtraTrees for larger datasets
        if self.X_train is not None and self.X_train.shape[0] > 100:
            models['ExtraTrees'] = ExtraTreesClassifier(random_state=42, n_jobs=-1)
            param_grids['ExtraTrees'] = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10, None],
            }
        
        return models, param_grids
    
    def _get_regression_models(self):
        """Get regression models and parameter grids"""
        models = {
            'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Ridge': Ridge(random_state=42),
            'KNN': KNeighborsRegressor(n_jobs=-1),
        }
        
        param_grids = {
            'RandomForest': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10, None],
            },
            'Ridge': {
                'model__alpha': [0.1, 1, 10, 100]
            },
            'KNN': {
                'model__n_neighbors': [3, 5, 7],
            }
        }
        
        # Add LinearSVR for larger datasets
        if self.X_train is not None and self.X_train.shape[0] > 100:
            models['LinearSVR'] = LinearSVR(random_state=42, max_iter=1000)
            param_grids['LinearSVR'] = {
                'model__C': [0.1, 1, 10],
                'model__epsilon': [0.1, 0.2]
            }
        
        # Add ExtraTrees for larger datasets
        if self.X_train is not None and self.X_train.shape[0] > 100:
            models['ExtraTrees'] = ExtraTreesRegressor(random_state=42, n_jobs=-1)
            param_grids['ExtraTrees'] = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10, None],
            }
        
        return models, param_grids

    def evaluate(self):
        if self.model is None:
            print("  Model not trained.")
            return
        
        if self.X_test is None or self.y_test is None:
            print("  Test data not available.")
            return
            
        try:
            y_pred = self.model.predict(self.X_test)
            
            if self.task_type == 'classification':
                acc = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                print(f"  Accuracy: {acc:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Best Model: {type(self.model.named_steps['model']).__name__}")
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, y_pred)
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R2 Score: {r2:.4f}")
                print(f"  Best Model: {type(self.model.named_steps['model']).__name__}")
        except Exception as e:
            print(f"  Error during evaluation: {e}")

    def _load_data(self, filepath):
        """
        Load data from a file. 
        Supports:
        1. Custom Sparse format (index:value without label at start) -> Returns CSR Matrix
        2. Standard CSV/Space separated -> Returns Numpy Array
        """
        try:
            if filepath.endswith('.data'):
                # Try sparse format first
                try:
                    data = []
                    indices = []
                    indptr = [0]
                    n_features = 0
                    n_lines = 0
                    
                    with open(filepath, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                                
                            items = line.split()
                            sparse_format = any(':' in item for item in items)
                            
                            if sparse_format:
                                for item in items:
                                    if ':' in item:
                                        idx_str, val_str = item.split(':')
                                        idx = int(idx_str) - 1  # 1-based to 0-based
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
                    
                    if len(data) > 0:
                        X = csr_matrix((data, indices, indptr), shape=(n_lines, n_features))
                        return X
                    else:
                        # Fallback to pandas for dense data
                        return pd.read_csv(filepath, sep=r'\s+', header=None).values
                        
                except Exception as sparse_error:
                    if self.debug:
                        print(f"  Sparse loading failed, trying dense: {sparse_error}")
                    # Fallback to dense format
                    return pd.read_csv(filepath, sep=r'\s+', header=None).values

            elif filepath.endswith('.solution'):
                try:
                    # Try reading as dense matrix
                    data = pd.read_csv(filepath, sep=r'\s+', header=None).values
                    # If single column, flatten it
                    if data.shape[1] == 1:
                        data = data.ravel()
                    return data
                except:
                    # Try reading line by line for sparse solutions (unlikely but possible)
                    with open(filepath, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    return np.array([float(val) for line in lines for val in line.split()])
                    
            else:
                return pd.read_csv(filepath, sep=r'\s+', header=None).values
                
        except Exception as e:
            if self.debug:
                print(f"Error loading {filepath}: {e}")
            return np.array([])

    def _load_types(self, filepath):
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                types = [line.strip() for line in f.readlines()]
            return types
        except:
            return None
