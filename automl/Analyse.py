import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from sklearn.datasets import make_classification, make_regression
import warnings

# Suppression des warnings pour une sortie plus propre
warnings.filterwarnings('ignore')

def generate_synthetic_dataset(name):
    """
    Génère des datasets synthétiques avec différentes caractéristiques
    pour simuler data_A à data_K
    """
    if name == 'data_A':
        # Petit dataset numérique dense
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        feature_types = ['Numerical'] * 5
        missing_rate = 0.0
        imbalance = 'balanced'
        
    elif name == 'data_B':
        # Dataset avec features catégorielles
        X_num, y = make_classification(n_samples=500, n_features=8, n_classes=3, random_state=42)
        # Ajouter des données catégorielles
        X_cat = np.random.choice(['A', 'B', 'C'], size=(500, 2))
        X = np.hstack([X_num, X_cat])
        feature_types = ['Numerical'] * 8 + ['Categorical'] * 2
        missing_rate = 0.05
        imbalance = 'moderate'
        
    elif name == 'data_C':
        # Dataset sparse
        X = np.random.rand(1000, 50)
        # Rendre sparse
        mask = np.random.rand(1000, 50) > 0.9
        X[~mask] = 0
        X = csr_matrix(X)
        y = np.random.randint(0, 2, 1000)
        feature_types = None
        missing_rate = 0.0
        imbalance = 'balanced'
        
    elif name == 'data_D':
        # Dataset de régression
        X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
        feature_types = ['Numerical'] * 10
        missing_rate = 0.1
        imbalance = 'N/A'
        
    elif name == 'data_E':
        # Dataset multi-classe déséquilibré
        X, y = make_classification(
            n_samples=300, 
            n_features=7, 
            n_classes=4,
            n_informative=5,
            weights=[0.5, 0.25, 0.15, 0.1],
            random_state=42
        )
        feature_types = ['Numerical'] * 7
        missing_rate = 0.02
        imbalance = 'high'
        
    elif name == 'data_F':
        # Dataset mixte avec valeurs manquantes
        X_num, y = make_classification(n_samples=150, n_features=6, random_state=42)
        X_cat = np.random.choice(['X', 'Y', 'Z'], size=(150, 3))
        X = np.hstack([X_num, X_cat])
        feature_types = ['Numerical'] * 6 + ['Categorical'] * 3
        missing_rate = 0.15
        imbalance = 'balanced'
        
    elif name == 'data_G':
        # Grande dataset sparse
        X = np.random.rand(5000, 100)
        mask = np.random.rand(5000, 100) > 0.95
        X[~mask] = 0
        X = csc_matrix(X)
        y = np.random.randint(0, 5, 5000)
        feature_types = None
        missing_rate = 0.0
        imbalance = 'moderate'
        
    elif name == 'data_H':
        # Dataset binaire déséquilibré
        X, y = make_classification(
            n_samples=400,
            n_features=12,
            n_classes=2,
            weights=[0.9, 0.1],
            random_state=42
        )
        feature_types = ['Numerical'] * 12
        missing_rate = 0.03
        imbalance = 'very high'
        
    elif name == 'data_I':
        # Dataset avec beaucoup de catégories
        X_num, y = make_regression(n_samples=250, n_features=4, random_state=42)
        X_cat = np.random.choice(['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5'], size=(250, 4))
        X = np.hstack([X_num, X_cat])
        feature_types = ['Numerical'] * 4 + ['Categorical'] * 4
        missing_rate = 0.08
        imbalance = 'N/A'
        
    elif name == 'data_J':
        # Très petit dataset
        X, y = make_classification(n_samples=50, n_features=3, random_state=42)
        feature_types = ['Numerical'] * 3
        missing_rate = 0.0
        imbalance = 'balanced'
        
    elif name == 'data_K':
        # Dataset avec mix de types et valeurs manquantes
        X_num, y = make_classification(n_samples=600, n_features=9, n_classes=3, random_state=42)
        X_cat = np.random.choice(['Low', 'Medium', 'High'], size=(600, 2))
        X = np.hstack([X_num, X_cat])
        
        # Ajouter des NaN
        n_missing = int(600 * 11 * 0.07)
        indices = np.random.choice(600 * 11, n_missing, replace=False)
        X_flat = X.flatten()
        X_flat[indices] = np.nan
        X = X_flat.reshape(600, 11)
        
        feature_types = ['Numerical'] * 9 + ['Categorical'] * 2
        missing_rate = 0.07
        imbalance = 'moderate'
    
    else:
        raise ValueError(f"Dataset {name} inconnu")
    
    return X, y, feature_types, missing_rate, imbalance

def analyze_dataset(X, y, feature_types, dataset_name):
    """
    Analyse un dataset et retourne ses caractéristiques
    """
    # Initialiser les valeurs
    n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
    n_features = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1
    
    # Vérifier si sparse
    is_sparse = issparse(X)
    
    # Calculer la sparsité
    if is_sparse:
        sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
    else:
        X_dense = np.array(X)
        sparsity = np.sum(X_dense == 0) / (n_samples * n_features) if n_samples * n_features > 0 else 0
    
    # Pourcentage de valeurs manquantes
    if is_sparse:
        missing_pct = 0.0  # Les matrices sparse n'ont généralement pas de NaN
    else:
        X_array = np.array(X)
        missing_pct = np.sum(np.isnan(X_array)) / (n_samples * n_features) * 100
    
    # Nombre de classes (pour classification)
    task_type = infer_task_type_simple(y)
    
    if task_type == 'classification':
        y_flat = np.array(y).ravel()
        unique_classes = np.unique(y_flat[~np.isnan(y_flat)])
        n_classes = len(unique_classes)
        
        # Calculer le déséquilibre
        if n_classes > 1:
            class_counts = [np.sum(y_flat == cls) for cls in unique_classes]
            imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
            
            if imbalance_ratio < 1.5:
                imbalance_level = 'balanced'
            elif imbalance_ratio < 3:
                imbalance_level = 'moderate'
            elif imbalance_ratio < 10:
                imbalance_level = 'high'
            else:
                imbalance_level = 'very high'
        else:
            imbalance_level = 'N/A'
    else:
        n_classes = 'N/A'
        imbalance_level = 'N/A'
    
    return {
        'Dataset': dataset_name,
        'n_samples': n_samples,
        'n_features': n_features,
        'Sparsité': f"{sparsity:.2%}",
        'Missing': f"{missing_pct:.1f}%",
        'n_classes': n_classes if task_type == 'classification' else 'N/A',
        'Imbalance': imbalance_level
    }

def infer_task_type_simple(y):
    """
    Version simplifiée de l'inférence du type de tâche
    """
    y_array = np.asarray(y)
    
    if y_array.size == 0:
        return 'regression'
    
    # Vérifier si multi-colonne (one-hot encoded)
    if hasattr(y_array, 'shape') and len(y_array.shape) == 2 and y_array.shape[1] > 1:
        return 'classification'
    
    # Analyser les valeurs uniques
    y_flat = y_array.ravel()
    valid_mask = ~np.isnan(y_flat)
    
    if not np.any(valid_mask):
        return 'regression'
    
    y_valid = y_flat[valid_mask]
    n_unique = len(np.unique(y_valid))
    
    # Heuristique simple
    if n_unique <= 20:
        # Vérifier si les valeurs sont principalement entières
        if np.all(np.equal(np.mod(y_valid, 1), 0)):
            return 'classification'
        # Vérifier pour la classification binaire
        if n_unique == 2:
            unique_vals = np.unique(y_valid)
            if set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({-1, 1}):
                return 'classification'
    
    return 'regression'

def main():
    """
    Fonction principale qui analyse tous les datasets
    """
    datasets = ['data_A', 'data_B', 'data_C', 'data_D', 'data_E', 
                'data_F', 'data_G', 'data_H', 'data_I', 'data_J', 'data_K']
    
    results = []
    
    print("Analyse des datasets...")
    print("-" * 80)
    
    for dataset_name in datasets:
        print(f"Traitement de {dataset_name}...")
        
        # Générer le dataset synthétique
        X, y, feature_types, missing_rate, expected_imbalance = generate_synthetic_dataset(dataset_name)
        
        # Analyser le dataset
        analysis = analyze_dataset(X, y, feature_types, dataset_name)
        
        # Ajouter les informations sur le pipeline
        is_sparse = issparse(X)
        n_features = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1
        
        # Obtenir le résumé du preprocessing
        preprocessing_summary = get_preprocessing_summary(feature_types, n_features, is_sparse)
        
        # Ajouter le type de pipeline
        if is_sparse:
            pipeline_type = "Sparse (MaxAbsScaler)"
        elif feature_types and any('Categorical' in str(ft) for ft in feature_types):
            pipeline_type = "Mixte (Numerical + Categorical)"
        else:
            pipeline_type = "Numerical standard"
        
        analysis['Pipeline Type'] = pipeline_type
        results.append(analysis)
    
    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    
    # Réorganiser les colonnes pour correspondre à l'image
    column_order = ['Dataset', 'n_samples', 'n_features', 'Sparsité', 
                    'Missing', 'n_classes', 'Imbalance', 'Pipeline Type']
    df_results = df_results[column_order]
    
    # Afficher le tableau
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES ANALYSES DES DATASETS")
    print("=" * 80)
    print("\n")
    
    # Afficher avec un formatage tabulaire
    from tabulate import tabulate
    print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))
    
    # Afficher sans tabulate si non disponible
    print("\nFormat simplifié:")
    print(df_results.to_string(index=False))
    
    # Sauvegarder dans un fichier CSV
    df_results.to_csv('dataset_analysis.csv', index=False)
    print(f"\nRésultats sauvegardés dans 'dataset_analysis.csv'")
    
    # Afficher quelques statistiques globales
    print("\n" + "=" * 80)
    print("STATISTIQUES GLOBALES")
    print("=" * 80)
    
    total_samples = df_results['n_samples'].sum()
    avg_features = df_results['n_features'].mean()
    num_classification = sum(df_results['n_classes'] != 'N/A')
    num_regression = len(datasets) - num_classification
    
    print(f"Total des échantillons (tous datasets): {total_samples:,}")
    print(f"Nombre moyen de features: {avg_features:.1f}")
    print(f"Datasets de classification: {num_classification}")
    print(f"Datasets de régression: {num_regression}")
    
    # Compter les types de pipelines
    pipeline_counts = df_results['Pipeline Type'].value_counts()
    print("\nDistribution des types de pipelines:")
    for pipeline_type, count in pipeline_counts.items():
        print(f"  {pipeline_type}: {count} dataset(s)")

if __name__ == "__main__":
    main()