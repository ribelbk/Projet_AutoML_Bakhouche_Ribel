"""
Models definition and parameter grids for AutoML.
"""
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings


def _get_classification_models(is_sparse=False, dataset_size=100, n_features=100, n_jobs=-1):
    """
    Get classification models and parameter grids.
    
    Parameters:
    -----------
    is_sparse : bool
        Whether the data is sparse (e.g., text data)
    dataset_size : int
        Number of samples in the dataset
    n_features : int
        Number of features in the dataset
    n_jobs : int
        Number of parallel jobs for models that support it
    
    Returns:
    --------
    models : dict
        Dictionary of model instances
    param_grids : dict
        Dictionary of parameter grids for each model
    """
    models = {}
    param_grids = {}
    
    # Détection de la taille du dataset
    too_many_features = n_features > 5000
    is_large_dataset = n_features > 10000 or dataset_size > 10000
    is_medium_dataset = n_features > 1000 or dataset_size > 5000
    
    # 1. CAS TRÈS VOLUMINEUX (large ou medium dataset)
    if is_large_dataset or is_medium_dataset:
        print(f"  [INFO] Dataset volumineux détecté: {dataset_size} échantillons, {n_features} features")
        
        # Pour éviter les problèmes de mémoire/parallélisme avec grands datasets
        safe_n_jobs = 1 if dataset_size > 5000 or n_features > 1000 else min(2, n_jobs)
        
        # A. LINEAR SVC - OPTIMAL POUR HAUTE DIMENSION
        models['LinearSVC'] = LinearSVC(
            random_state=42,
            dual='auto',  # Auto-select du meilleur algorithme
            max_iter=1000,
            tol=1e-3,  # Tolérance augmentée pour vitesse
            verbose=0
        )
        param_grids['LinearSVC'] = {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l2'],
            'model__loss': ['squared_hinge'],
        }
        
        # B. RIDGE CLASSIFIER - ALTERNATIVE RAPIDE À LOGISTIC REGRESSION
        models['RidgeClassifier'] = RidgeClassifier(
            random_state=42,
            solver='auto'  # Choix automatique du solveur optimal
        )
        param_grids['RidgeClassifier'] = {
            'model__alpha': [0.1, 1, 10, 100],
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
        
        # C. SGDCLASSIFIER - SCALABLE POUR TRÈS GRANDS DATASETS
        if dataset_size > 5000 or n_features > 5000:
            models['SGDClassifier'] = SGDClassifier(
                random_state=42,
                max_iter=1000,
                tol=1e-3,
                early_stopping=True,
                n_iter_no_change=5,
                validation_fraction=0.1,
                verbose=0
            )
            param_grids['SGDClassifier'] = {
                'model__loss': ['log_loss', 'hinge', 'modified_huber'],
                'model__penalty': ['l2', 'l1', 'elasticnet'],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__l1_ratio': [0, 0.15, 0.5, 0.85, 1],
                'model__learning_rate': ['optimal', 'adaptive']
            }
        
        # D. LOGISTIC REGRESSION - AVEC CONFIGURATION OPTIMISÉE
        if not too_many_features and n_features <= 10000:
            # Choix du solver optimal selon les caractéristiques
            if is_sparse:
                solver = 'saga'  # Meilleur pour données sparse
            elif n_features > dataset_size:
                solver = 'liblinear'  # Meilleur quand n_features > n_samples
            elif dataset_size > 10000:
                solver = 'saga'  # Scalable pour très grands datasets
            else:
                solver = 'lbfgs'  # Rapide pour datasets moyens
            
            max_iter_val = 300 if n_features > 1000 else 500
            
            models['LogisticRegression'] = LogisticRegression(
                random_state=42,
                max_iter=max_iter_val,
                solver=solver,
                n_jobs=safe_n_jobs,
                tol=1e-3 if solver != 'liblinear' else 1e-4,
                verbose=0
            )
            
            # Configuration des paramètres selon le solver
            if solver == 'liblinear':
                param_grids['LogisticRegression'] = {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l2'],
                }
            elif solver == 'saga':
                param_grids['LogisticRegression'] = {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l2', 'elasticnet'],
                    'model__l1_ratio': [0, 0.5, 1] if n_features > 1000 else [0.5]
                }
            else:  # lbfgs ou newton-cg
                param_grids['LogisticRegression'] = {
                    'model__C': [0.1, 1, 10],
                }
        
        # E. MODÈLES LÉGERS SUPPLÉMENTAIRES
        # LDA avec shrinkage pour haute dimension
        if n_features > 1000 and not is_sparse and dataset_size < 100000:
            models['LDA'] = LinearDiscriminantAnalysis(
                solver='lsqr',
                shrinkage='auto'
            )
            param_grids['LDA'] = {
                'model__shrinkage': [None, 'auto', 0.5, 0.9],
                'model__solver': ['lsqr', 'eigen']
            }
        
        # Naive Bayes selon le type de données
        if is_sparse:
            models['MultinomialNB'] = MultinomialNB()
            param_grids['MultinomialNB'] = {
                'model__alpha': [0.1, 0.5, 1.0],
                'model__fit_prior': [True, False]
            }
        elif n_features < 1000:
            models['GaussianNB'] = GaussianNB()
            param_grids['GaussianNB'] = {
                'model__var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        
        # F. ARBRE SIMPLE SI PAS TROP GRAND
        if n_features < 50000 and dataset_size < 50000:
            models['DecisionTree'] = DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            )
            param_grids['DecisionTree'] = {
                'model__max_depth': [5, 10, 15],
                'model__min_samples_split': [10, 20, 30],
                'model__min_samples_leaf': [5, 10, 20]
            }
        
        return models, param_grids
    
    # 2. CAS NORMAL (petit à moyen dataset)
    # RandomForest - avec vérifications
    if n_features < 10000 and dataset_size < 100000:
        models['RandomForest'] = RandomForestClassifier(
            random_state=42,
            n_jobs=n_jobs,
            verbose=0
        )
        param_grids['RandomForest'] = {
            'model__n_estimators': [50, 100] if dataset_size > 500 else [50],
            'model__max_depth': [5, 10, None] if n_features < 500 else [5, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', None]
        }
    
    # LogisticRegression - toujours inclus si features raisonnables
    if not too_many_features:
        models['LogisticRegression'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=n_jobs if dataset_size > 100 else 1,
            solver='lbfgs' if not is_sparse else 'saga'
        )
        param_grids['LogisticRegression'] = {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l2'],
        }
    
    # KNN - seulement pour datasets petits
    if n_features < 1000 and dataset_size < 5000:
        models['KNN'] = KNeighborsClassifier(
            n_jobs=n_jobs if dataset_size < 1000 else 1
        )
        param_grids['KNN'] = {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan', 'minkowski']
        }
    
    # Modèles selon le type de données
    if is_sparse:
        if not too_many_features:
            models['LinearSVC'] = LinearSVC(
                random_state=42,
                dual=False,
                max_iter=1000
            )
            param_grids['LinearSVC'] = {
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l2'],
            }
        
        models['MultinomialNB'] = MultinomialNB()
        param_grids['MultinomialNB'] = {
            'model__alpha': [0.1, 0.5, 1.0, 2.0],
        }
    else:
        if n_features < 1000:
            models['GaussianNB'] = GaussianNB()
            param_grids['GaussianNB'] = {
                'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        
        if n_features < 1000 and dataset_size > n_features:
            models['LDA'] = LinearDiscriminantAnalysis()
            param_grids['LDA'] = {
                'model__solver': ['svd', 'lsqr', 'eigen'],
                'model__shrinkage': [None, 'auto']
            }
    
    # Tree-based models
    if dataset_size > 50 and n_features < 10000:
        models['DecisionTree'] = DecisionTreeClassifier(random_state=42)
        param_grids['DecisionTree'] = {
            'model__max_depth': [3, 5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10, 20],
            'model__min_samples_leaf': [1, 2, 5, 10],
            'model__criterion': ['gini', 'entropy']
        }
    
    # Ensemble models
    if dataset_size > 100 and n_features < 5000:
        models['ExtraTrees'] = ExtraTreesClassifier(
            random_state=42,
            n_jobs=n_jobs if dataset_size > 500 else 1
        )
        param_grids['ExtraTrees'] = {
            'model__n_estimators': [50, 100],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5, 10],
        }
        
        if dataset_size < 10000 and n_features < 1000:
            models['GradientBoosting'] = GradientBoostingClassifier(
                random_state=42,
                n_iter_no_change=5,
                validation_fraction=0.1
            )
            param_grids['GradientBoosting'] = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 1.0]
            }
    
    # FALLBACK SI AUCUN MODÈLE
    if not models:
        print(f"  [WARNING] Aucun modèle sélectionné, utilisation des modèles de fallback")
        
        # Ridge Classifier comme fallback principal
        models['RidgeClassifier'] = RidgeClassifier(
            random_state=42,
            solver='auto'
        )
        param_grids['RidgeClassifier'] = {
            'model__alpha': [0.1, 1, 10],
        }
        
        # Decision Tree comme deuxième fallback
        models['DecisionTree'] = DecisionTreeClassifier(
            random_state=42,
            max_depth=10
        )
        param_grids['DecisionTree'] = {
            'model__max_depth': [5, 10],
            'model__min_samples_split': [2, 10],
        }
        
        # Naive Bayes selon type
        if is_sparse:
            models['MultinomialNB'] = MultinomialNB()
            param_grids['MultinomialNB'] = {
                'model__alpha': [0.1, 1.0],
            }
    
    return models, param_grids


def _get_regression_models(dataset_size=100, n_features=100, n_jobs=-1):
    """
    Get regression models and parameter grids.
    
    Parameters:
    -----------
    dataset_size : int
        Number of samples in the dataset
    n_features : int
        Number of features in the dataset
    n_jobs : int
        Number of parallel jobs for models that support it
    
    Returns:
    --------
    models : dict
        Dictionary of model instances
    param_grids : dict
        Dictionary of parameter grids for each model
    """
    models = {}
    param_grids = {}
    
    # Détection de la taille
    too_many_features = n_features > 5000
    is_large_dataset = n_features > 10000 or dataset_size > 10000
    is_medium_dataset = n_features > 1000 or dataset_size > 5000
    
    # 1. CAS VOLUMINEUX
    if is_large_dataset or is_medium_dataset:
        print(f"  [INFO] Dataset volumineux détecté: {dataset_size} échantillons, {n_features} features")
        
        safe_n_jobs = 1 if dataset_size > 5000 or n_features > 1000 else min(2, n_jobs)
        
        # A. LINEAR SVR - RAPIDE ET SCALABLE
        models['LinearSVR'] = LinearSVR(
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            verbose=0
        )
        param_grids['LinearSVR'] = {
            'model__C': [0.1, 1, 10],
            'model__epsilon': [0.1, 0.2, 0.5],
            'model__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
        }
        
        # B. RIDGE REGRESSION - STABLE POUR HAUTE DIMENSION
        models['Ridge'] = Ridge(random_state=42)
        param_grids['Ridge'] = {
            'model__alpha': [0.1, 1, 10, 100, 1000],
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
        
        # C. SGD REGRESSOR - POUR TRÈS GRANDS DATASETS
        if dataset_size > 5000 or n_features > 5000:
            models['SGDRegressor'] = SGDRegressor(
                random_state=42,
                max_iter=1000,
                tol=1e-3,
                early_stopping=True,
                n_iter_no_change=5,
                verbose=0
            )
            param_grids['SGDRegressor'] = {
                'model__loss': ['squared_error', 'huber', 'epsilon_insensitive'],
                'model__penalty': ['l2', 'l1', 'elasticnet'],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__l1_ratio': [0, 0.15, 0.5, 0.85, 1],
                'model__learning_rate': ['optimal', 'invscaling', 'adaptive']
            }
        
        # D. LASSO - SÉLECTION DE FEATURES
        if n_features > 1000 and dataset_size > 100:
            models['Lasso'] = Lasso(
                random_state=42,
                max_iter=1000,
                selection='random' if n_features > 5000 else 'cyclic',
                tol=1e-3
            )
            param_grids['Lasso'] = {
                'model__alpha': [0.001, 0.01, 0.1, 1],
                'model__max_iter': [1000, 2000]
            }
        
        # E. MODÈLES SUPPLÉMENTAIRES
        # ElasticNet si dataset pas trop grand
        if n_features < 20000 and dataset_size < 50000:
            models['ElasticNet'] = ElasticNet(
                random_state=42,
                max_iter=1000,
                selection='random' if n_features > 1000 else 'cyclic'
            )
            param_grids['ElasticNet'] = {
                'model__alpha': [0.001, 0.01, 0.1, 1],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        
        return models, param_grids
    
    # 2. CAS NORMAL
    # RandomForest
    if n_features < 10000:
        models['RandomForest'] = RandomForestRegressor(
            random_state=42,
            n_jobs=n_jobs
        )
        param_grids['RandomForest'] = {
            'model__n_estimators': [50, 100, 200] if dataset_size > 1000 else [50, 100],
            'model__max_depth': [5, 10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', None]
        }
    
    # Ridge Regression
    models['Ridge'] = Ridge(random_state=42)
    param_grids['Ridge'] = {
        'model__alpha': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['auto', 'svd', 'cholesky']
    }
    
    # KNN
    if n_features < 1000 and dataset_size < 5000:
        models['KNN'] = KNeighborsRegressor(n_jobs=n_jobs)
        param_grids['KNN'] = {
            'model__n_neighbors': [3, 5, 7, 9, 11],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan', 'minkowski']
        }
    
    # LinearSVR
    models['LinearSVR'] = LinearSVR(
        random_state=42,
        max_iter=5000
    )
    param_grids['LinearSVR'] = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__epsilon': [0.1, 0.2, 0.5],
    }
    
    # Lasso
    models['Lasso'] = Lasso(
        random_state=42,
        max_iter=5000
    )
    param_grids['Lasso'] = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10],
        'model__selection': ['cyclic', 'random']
    }
    
    # Tree-based models
    if dataset_size > 50 and n_features < 10000:
        models['DecisionTree'] = DecisionTreeRegressor(random_state=42)
        param_grids['DecisionTree'] = {
            'model__max_depth': [3, 5, 10, 15, 20, None],
            'model__min_samples_split': [2, 5, 10, 20],
            'model__min_samples_leaf': [1, 2, 5, 10],
            'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error']
        }
    
    # Ensemble models
    if dataset_size > 100 and n_features < 5000:
        models['ExtraTrees'] = ExtraTreesRegressor(
            random_state=42,
            n_jobs=n_jobs
        )
        param_grids['ExtraTrees'] = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, 20, None],
            'model__min_samples_split': [2, 5, 10],
        }
        
        if dataset_size < 10000 and n_features < 1000:
            models['GradientBoosting'] = GradientBoostingRegressor(
                random_state=42,
                n_iter_no_change=5
            )
            param_grids['GradientBoosting'] = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0]
            }
    
    # ElasticNet
    if dataset_size > 200 and n_features < 10000:
        models['ElasticNet'] = ElasticNet(
            random_state=42,
            max_iter=5000
        )
        param_grids['ElasticNet'] = {
            'model__alpha': [0.001, 0.01, 0.1, 1, 10],
            'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    
    # FALLBACK
    if not models:
        print(f"  [WARNING] Aucun modèle de régression sélectionné, fallback aux modèles de base")
        
        models['Ridge'] = Ridge(random_state=42)
        param_grids['Ridge'] = {'model__alpha': [0.1, 1, 10]}
        
        models['LinearSVR'] = LinearSVR(random_state=42, max_iter=1000)
        param_grids['LinearSVR'] = {'model__C': [0.1, 1, 10]}
        
        models['DecisionTree'] = DecisionTreeRegressor(random_state=42)
        param_grids['DecisionTree'] = {'model__max_depth': [5, 10, 15]}
    
    return models, param_grids


def get_scoring_metric(task_type):
    """
    Get appropriate scoring metric for the task type.
    
    Parameters:
    -----------
    task_type : str
        'classification' or 'regression'
    
    Returns:
    --------
    scoring : str
        Scoring metric name
    """
    if task_type == 'classification':
        return 'accuracy'
    elif task_type == 'regression':
        return 'neg_mean_squared_error'
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_cv_folds(n_samples):
    """
    Determine appropriate number of CV folds based on dataset size.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples in training set
    
    Returns:
    --------
    cv_folds : int
        Number of cross-validation folds
    """
    if n_samples < 20:
        return 2  # Simple train-test split
    elif n_samples < 50:
        return 3
    elif n_samples < 100:
        return 4
    elif n_samples < 1000:
        return 5
    else:
        return 3  # Pour très grands datasets, utiliser moins de folds pour vitesse


def optimize_for_large_datasets(estimator, X, y, task_type='classification', max_features=1000):
    """
    Fonction utilitaire pour optimiser les modèles sur grands datasets.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        Modèle à optimiser
    X : array-like
        Features
    y : array-like
        Target
    task_type : str
        Type de tâche
    max_features : int
        Nombre maximum de features à conserver
    
    Returns:
    --------
    optimized_estimator : Pipeline
        Pipeline optimisée
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Créer une pipeline avec sélection de features si nécessaire
    steps = []
    
    # StandardScaler pour les modèles linéaires
    if task_type in ['classification', 'regression'] and not hasattr(estimator, 'feature_importances_'):
        steps.append(('scaler', StandardScaler(with_mean=not hasattr(X, 'toarray'))))
    
    # Sélection de features si trop de features
    if X.shape[1] > max_features and task_type == 'classification':
        from sklearn.feature_selection import SelectKBest, f_classif
        steps.append(('feature_selection', SelectKBest(f_classif, k=min(max_features, X.shape[1]))))
    elif X.shape[1] > max_features and task_type == 'regression':
        from sklearn.feature_selection import SelectKBest, f_regression
        steps.append(('feature_selection', SelectKBest(f_regression, k=min(max_features, X.shape[1]))))
    
    # Ajouter l'estimateur
    steps.append(('model', estimator))
    
    return Pipeline(steps)


def get_timeout_settings(n_samples, n_features):
    """
    Get timeout settings based on dataset characteristics.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    
    Returns:
    --------
    timeout_per_model : int
        Timeout in seconds per model
    """
    complexity = n_samples * n_features
    
    if complexity < 10000:  # Très petit
        return 60
    elif complexity < 100000:  # Petit
        return 120
    elif complexity < 1000000:  # Moyen
        return 180
    elif complexity < 10000000:  # Grand
        return 300
    else:  # Très grand
        return 600  # 10 minutes maximum


def select_best_models_for_task(task_type, is_sparse, dataset_size, n_features):
    """
    Sélection intelligente des meilleurs modèles pour la tâche.
    
    Parameters:
    -----------
    task_type : str
        Type de tâche
    is_sparse : bool
        Si les données sont sparse
    dataset_size : int
        Taille du dataset
    n_features : int
        Nombre de features
    
    Returns:
    --------
    selected_models : list
        Liste des noms des modèles sélectionnés
    """
    # Prioriser les modèles selon les caractéristiques
    if task_type == 'classification':
        if is_sparse:
            # Pour données sparse (texte)
            priority_models = ['LinearSVC', 'SGDClassifier', 'MultinomialNB', 'RidgeClassifier']
            if n_features < 10000:
                priority_models.append('LogisticRegression')
        elif n_features > 10000:
            # Pour très haute dimension
            priority_models = ['LinearSVC', 'RidgeClassifier', 'SGDClassifier', 'LDA']
        elif dataset_size > 10000:
            # Pour grands datasets
            priority_models = ['LinearSVC', 'RidgeClassifier', 'RandomForest', 'SGDClassifier']
        else:
            # Cas général
            priority_models = ['RandomForest', 'LogisticRegression', 'GradientBoosting', 'ExtraTrees']
    else:  # regression
        if n_features > 10000:
            priority_models = ['LinearSVR', 'Ridge', 'SGDRegressor', 'Lasso']
        elif dataset_size > 10000:
            priority_models = ['LinearSVR', 'Ridge', 'RandomForest', 'SGDRegressor']
        else:
            priority_models = ['RandomForest', 'GradientBoosting', 'Ridge', 'ElasticNet']
    
    return priority_models