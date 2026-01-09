"""
Main AutoML core module.
"""
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

# Imports locaux
from .preprocessing import build_pipeline, infer_task_type
from .models import _get_classification_models, _get_regression_models, get_scoring_metric, get_cv_folds
from .metrics import classification_metrics, regression_metrics
from .utils import load_data, load_types, setup_debug_mode, check_required_files
from scipy.sparse import issparse

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Global variable to store trained predictors
trained_predictors = {}


def fit(data_dest, dataset_name=None, debug=False):
    """
    Train models on all datasets found in data_dest.
    
    Parameters:
    -----------
    data_dest : str
        Directory containing datasets
    dataset_name : str, optional
        Specific dataset to train on
    debug : bool
        Enable debug mode
    """
    global trained_predictors
    trained_predictors = {}
    
    setup_debug_mode(debug)
    
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

    for dataset_idx, subdir in enumerate(subdirs):
        dataset_path = os.path.join(data_dest, subdir)
        dataset_name = subdir
        
        # Check for required files
        if not check_required_files(dataset_path, dataset_name, debug):
            if debug:
                print(f"Skipping {dataset_name}: missing required files")
            continue

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


def eval():
    """
    Evaluate trained models and print summary.
    """
    global trained_predictors
    if not trained_predictors:
        print("No models trained. Run automl.fit() first.")
        return
    
    print("\n" + "="*60)
    print("                 PERFORMANCE SUMMARY                 ")
    print("="*60)
    
    for name, predictor in trained_predictors.items():
        print(f"\n{'='*50}")
        print(f"DATASET: {name}")
        print(f"{'='*50}")
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
        self.task_type = None 
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.all_models_results = {}  # Stocke les résultats de tous les modèles
        self.best_model_name = None   # Nom du meilleur modèle
        
    def fit(self):
        """Train the AutoML predictor on the dataset."""
        try:
            # Load and prepare data
            X, y = self._load_and_prepare_data()
            
            # Split data
            self._split_data(X, y)
            
            # Build preprocessing pipeline
            pipeline = self._build_pipeline()
            
            # Get models and train
            self._train_models(pipeline)
            
        except Exception as e:
            error_msg = f"Error in fit() for {self.name}: {str(e)}"
            if self.debug:
                import traceback
                traceback.print_exc()
            raise Exception(error_msg)
    
    def _load_and_prepare_data(self):
        """Load and prepare the dataset."""
        # Load data
        X = load_data(os.path.join(self.path, f"{self.name}.data"), self.debug)
        y = load_data(os.path.join(self.path, f"{self.name}.solution"), self.debug)
        
        if self.debug:
            print(f"  X shape: {X.shape if hasattr(X, 'shape') else 'N/A'}, "
                  f"type: {type(X).__name__}")
            print(f"  y shape: {y.shape if hasattr(y, 'shape') else 'N/A'}, "
                  f"type: {type(y).__name__}")
        
        # Validate data
        if (not hasattr(X, 'shape') or X.shape[0] == 0 or 
            not hasattr(y, 'shape') or y.shape[0] == 0):
            raise ValueError(f"Empty data or solution file for {self.name}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched samples: X has {X.shape[0]}, y has {y.shape[0]}")
        
        # Infer task type
        self.task_type = infer_task_type(y)
        
        if self.debug:
            print(f"  Task type: {self.task_type}")
            print(f"  Is sparse: {issparse(X)}")
        
        # Prepare target based on task type
        y = self._prepare_target(y)
        
        return X, y
    
    def _prepare_target(self, y):
        """Prepare target variable based on task type."""
        y = np.asarray(y)
        
        if self.task_type == 'classification':
            if y.ndim > 1 and y.shape[1] > 1:
                # One-hot encoded to labels
                y = np.argmax(y, axis=1)
            elif y.ndim > 1 and y.shape[1] == 1:
                # 2D single column to 1D
                y = y.ravel()
            
            # Convert to integer for classification
            return y.astype(int)
        else:
            # Regression - ensure numeric type
            if y.ndim > 1 and y.shape[1] == 1:
                y = y.ravel()
            return y.astype(float)
    
    def _split_data(self, X, y):
        """Split data into train and test sets."""
        test_size = 0.2  # 20% for test
        
        stratify = y if self.task_type == 'classification' and len(np.unique(y)) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        if self.debug:
            print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    def _build_pipeline(self):
        """Build the preprocessing pipeline."""
        # Load feature types
        type_file = os.path.join(self.path, f"{self.name}.type")
        feature_types = load_types(type_file)
        
        # Build pipeline
        is_sparse_data = issparse(self.X_train)
        return build_pipeline(feature_types, self.X_train.shape[1], is_sparse=is_sparse_data)
    
    def _train_models(self, pipeline):
        """Train and select the best model."""
        # Get appropriate models avec n_features
        if self.task_type == 'classification':
            models, param_grids = _get_classification_models(
                is_sparse=issparse(self.X_train),
                dataset_size=self.X_train.shape[0],
                n_features=self.X_train.shape[1]  # Nouveau paramètre
            )
        else:
            models, param_grids = _get_regression_models(
                dataset_size=self.X_train.shape[0],
                n_features=self.X_train.shape[1]  # Nouveau paramètre
            )
        
        # Get scoring metric and CV folds
        scoring = get_scoring_metric(self.task_type)
        cv_folds = get_cv_folds(self.X_train.shape[0])
        
        if self.debug:
            print(f"  Available models: {list(models.keys())}")
            print(f"  Using {cv_folds} CV folds, scoring: {scoring}")
        
        # Initialize dictionary to store all results
        self.all_models_results = {}
        best_score = -np.inf
        best_model = None
        best_model_name = None
        
        # Créer une barre de progression pour l'entraînement des modèles
        from tqdm import tqdm
        
        # Barre de progression principale pour les modèles
        pbar_models = tqdm(
            models.items(), 
            desc=f"{self.name:15s}",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for model_name, model in pbar_models:
            try:
                # Mettre à jour la description avec le modèle en cours
                pbar_models.set_description(f"{self.name:15s} - {model_name[:15]:15s}")
                
                if self.debug:
                    print(f"    Training {model_name}...")
                
                full_pipeline = Pipeline(steps=[('preprocessor', pipeline), ('model', model)])
                
                # Use GridSearchCV
                search = GridSearchCV(
                    full_pipeline,
                    param_grids[model_name],
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                
                # Créer une barre de progression pour la validation croisée
                if self.debug:
                    print(f"      Starting cross-validation ({cv_folds} folds)...")
                
                search.fit(self.X_train, self.y_train)
                cv_score = search.best_score_
                model_trained = search.best_estimator_
                best_params = search.best_params_
                
                # Store results for this model
                self.all_models_results[model_name] = {
                    'cv_score': cv_score,
                    'model': model_trained,
                    'params': best_params,
                    'model_type': type(model).__name__
                }
                
                # Check if this is the best model
                if cv_score > best_score:
                    best_score = cv_score
                    best_model = model_trained
                    best_model_name = model_name
                    
                    # Mettre à jour la barre avec le meilleur score
                    pbar_models.set_postfix({'best': f"{best_model_name[:10]}:{best_score:.3f}"})
                    
                    if self.debug:
                        print(f"      New best: {model_name} (CV score: {cv_score:.4f})")
                        
            except Exception as e:
                if self.debug:
                    print(f"    Error with {model_name}: {str(e)}")
                # Store failed model result
                self.all_models_results[model_name] = {
                    'cv_score': -np.inf,
                    'error': str(e),
                    'model_type': type(model).__name__
                }
                # Mettre à jour la barre avec l'erreur
                pbar_models.set_postfix_str(f"❌ {str(e)[:20]}...")
        
        # Fermer la barre de progression
        pbar_models.close()
        
        if best_model is None:
            raise ValueError("No model could be trained successfully")
        
        self.model = best_model
        self.best_model_name = best_model_name
        
        if self.debug:
            print(f"  Best model: {best_model_name} (CV score: {best_score:.4f})")
        else:
            # Afficher un résumé concis
            print(f"✓ {self.name:15s} : {best_model_name:20s} (CV: {best_score:.4f})")
    
    def evaluate(self):
        """Evaluate the trained model."""
        if self.model is None:
            print("  Model not trained.")
            return
        
        if self.X_test is None or self.y_test is None:
            print("  Test data not available.")
            return
        
        try:
            # Évaluer tous les modèles
            print("\n   Performance de tous les modèles:")
            print("  " + "-" * 50)
            
            results_summary = []
            
            for model_name, result in self.all_models_results.items():
                if 'error' in result:
                    # Modèle qui a échoué
                    print(f"  ❌ {model_name:20s} : ÉCHEC - {result['error']}")
                    continue
                
                # Prédire avec ce modèle
                model_pipeline = result['model']
                y_pred = model_pipeline.predict(self.X_test)
                
                # Calculer les métriques
                if self.task_type == 'classification':
                    metrics = classification_metrics(self.y_test, y_pred)
                    test_score = metrics['accuracy']
                    
                    results_summary.append({
                        'model': model_name,
                        'cv_score': result['cv_score'],
                        'test_accuracy': test_score,
                        'test_f1': metrics['f1_weighted'],
                        'params': result.get('params', {})
                    })
                    
                    print(f"   {model_name:20s} : "
                          f"CV={result['cv_score']:.4f} | "
                          f"Test Acc={test_score:.4f} | "
                          f"F1={metrics['f1_weighted']:.4f}")
                    
                else:
                    metrics = regression_metrics(self.y_test, y_pred)
                    
                    results_summary.append({
                        'model': model_name,
                        'cv_score': result['cv_score'],
                        'test_rmse': metrics['rmse'],
                        'test_r2': metrics['r2'],
                        'params': result.get('params', {})
                    })
                    
                    print(f"   {model_name:20s} : "
                          f"CV={result['cv_score']:.4f} | "
                          f"Test RMSE={metrics['rmse']:.4f} | "
                          f"R²={metrics['r2']:.4f}")
            
            # Afficher le meilleur modèle
            print("\n  " + "="*50)
            print("   MEILLEUR MODÈLE:")
            print("  " + "="*50)
            
            # Prédire avec le meilleur modèle
            y_pred_best = self.model.predict(self.X_test)
            
            if self.task_type == 'classification':
                metrics = classification_metrics(self.y_test, y_pred_best)
                print(f"  Modèle: {self.best_model_name}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}")
            else:
                metrics = regression_metrics(self.y_test, y_pred_best)
                print(f"  Modèle: {self.best_model_name}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R² Score: {metrics['r2']:.4f}")
                
        except Exception as e:
            print(f"  Error during evaluation: {e}")