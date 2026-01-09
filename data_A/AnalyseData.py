import numpy as np
import pandas as pd
import scipy.sparse
from collections import Counter
import os

def analyze_dataset(data_path, solution_path=None):
    """
    Analyse un dataset et retourne ses caractéristiques statistiques
    
    Paramètres:
    -----------
    data_path : str
        Chemin vers le fichier de données (data_A.data)
    solution_path : str, optionnel
        Chemin vers le fichier de solution (data_A.solution)
    
    Retourne:
    ---------
    dict : Dictionnaire contenant toutes les caractéristiques statistiques
    """
    
    print(f"Analyse du dataset: {data_path}")
    
    # Initialisation du dictionnaire de résultats
    stats = {
        'Dataset': os.path.basename(data_path).split('.')[0],
        'n_samples': 0,
        'n_features': 0,
        'Sparsité': 0.0,
        'Missing': 0.0,
        'n_classes': 0,
        'Imbalance': 0.0
    }
    
    try:
        # Essayer de lire le fichier comme une matrice sparse d'abord
        try:
            # Pour les formats de matrice sparse (comme libsvm)
            from sklearn.datasets import load_svmlight_file
            X, y = load_svmlight_file(data_path)
            is_sparse = True
        except:
            # Si ce n'est pas au format libsvm, essayer de lire comme CSV/TSV
            try:
                # Essayer différentes délimitations
                for delimiter in [',', '\t', ';', ' ']:
                    try:
                        df = pd.read_csv(data_path, delimiter=delimiter, header=None)
                        if df.shape[1] > 1:  # Au moins 2 colonnes
                            break
                    except:
                        continue
                
                # Si on a une solution séparée
                if solution_path and os.path.exists(solution_path):
                    y_df = pd.read_csv(solution_path, header=None)
                    y = y_df.values.flatten()
                    X = df.values
                else:
                    # La dernière colonne est probablement la target
                    X = df.iloc[:, :-1].values
                    y = df.iloc[:, -1].values
                
                is_sparse = False
            except Exception as e:
                print(f"Erreur de lecture: {e}")
                return stats
        
        # 1. Nombre d'échantillons
        stats['n_samples'] = X.shape[0]
        
        # 2. Nombre de features
        stats['n_features'] = X.shape[1]
        
        # 3. Calcul de la sparsité
        if is_sparse:
            # Pour les matrices sparse
            total_elements = X.shape[0] * X.shape[1]
            nonzero_elements = X.count_nonzero()
            stats['Sparsité'] = 1.0 - (nonzero_elements / total_elements)
        else:
            # Pour les matrices denses
            zero_elements = np.sum(X == 0)
            total_elements = X.size
            stats['Sparsité'] = zero_elements / total_elements
        
        # 4. Calcul des valeurs manquantes
        if not is_sparse:
            # Convertir en DataFrame pour faciliter la détection des NaN
            X_df = pd.DataFrame(X)
            missing_count = X_df.isnull().sum().sum()
            stats['Missing'] = missing_count / total_elements
        else:
            # Les matrices sparse ne supportent généralement pas les NaN
            stats['Missing'] = 0.0
        
        # 5. Nombre de classes
        if solution_path and os.path.exists(solution_path):
            # Lire les labels depuis le fichier solution
            try:
                y_df = pd.read_csv(solution_path, header=None)
                y = y_df.values.flatten()
            except:
                pass
        
        # Nettoyer les labels (supprimer les NaN)
        y_clean = y[~np.isnan(y)]
        
        # Compter les classes uniques
        unique_classes = np.unique(y_clean)
        stats['n_classes'] = len(unique_classes)
        
        # 6. Calcul de l'Imbalance
        if len(y_clean) > 0:
            class_counts = Counter(y_clean)
            if len(class_counts) > 1:
                # Calcul de l'imbalance ratio
                counts = list(class_counts.values())
                imbalance_ratio = max(counts) / min(counts)
                # Normaliser entre 0 et 1
                stats['Imbalance'] = 1.0 - (1.0 / imbalance_ratio)
            else:
                stats['Imbalance'] = 0.0  # Une seule classe
        
        # Formater les pourcentages
        stats['Sparsité'] = round(stats['Sparsité'] * 100, 2)
        stats['Missing'] = round(stats['Missing'] * 100, 2)
        stats['Imbalance'] = round(stats['Imbalance'], 4)
        
        return stats
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        return stats

def print_statistics_table(stats_dict):
    """
    Affiche les statistiques sous forme de tableau
    """
    print("\n" + "="*80)
    print("CARACTÉRISTIQUES STATISTIQUES DU DATASET")
    print("="*80)
    
    headers = ["Dataset", "n_samples", "n_features", "Sparsité (%)", 
               "Missing (%)", "n_classes", "Imbalance"]
    
    # Afficher l'en-tête
    header_row = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
    
    print(header_row)
    print(separator)
    
    # Afficher les données
    data_row = f"| {stats_dict['Dataset']:^10} | {stats_dict['n_samples']:^9} | "
    data_row += f"{stats_dict['n_features']:^11} | {stats_dict['Sparsité']:^13} | "
    data_row += f"{stats_dict['Missing']:^11} | {stats_dict['n_classes']:^10} | "
    data_row += f"{stats_dict['Imbalance']:^10} |"
    
    print(data_row)
    print("="*80)

# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Chemins vers vos fichiers
    data_file = "data_A.data"
    solution_file = "data_A.solution"
    
    # Vérifier si les fichiers existent
    if not os.path.exists(data_file):
        print(f"Fichier {data_file} non trouvé!")
        print("Recherche dans le répertoire courant...")
        files = os.listdir('.')
        print("Fichiers disponibles:", files)
    else:
        # Analyser le dataset
        stats = analyze_dataset(data_file, solution_file)
        
        # Afficher les résultats
        print_statistics_table(stats)
        
        # Afficher un résumé détaillé
        print("\n" + "-"*50)
        print("RÉSUMÉ DÉTAILLÉ:")
        print("-"*50)
        print(f"1. Nombre d'échantillons: {stats['n_samples']}")
        print(f"2. Nombre de features: {stats['n_features']}")
        print(f"3. Sparsité: {stats['Sparsité']}% (pourcentage de zéros)")
        print(f"4. Valeurs manquantes: {stats['Missing']}%")
        print(f"5. Nombre de classes: {stats['n_classes']}")
        print(f"6. Score d'Imbalance: {stats['Imbalance']:.4f}")
        print("   (0 = parfaitement équilibré, 1 = très déséquilibré)")