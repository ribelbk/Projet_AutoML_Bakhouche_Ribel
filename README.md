# AutoML Beta Library

Bienvenue dans la bibliothèque **AutoML Beta**. Ce projet propose un pipeline de Machine Learning Automatisé (AutoML) capable de traiter divers ensembles de données, d'entraîner plusieurs modèles (classification et régression) et de sélectionner le plus performant.

## 📌 Fonctionnalités

*   **Détection automatique** du type de tâche (Classification ou Régression).
*   **Support des données creuses (Sparse Matrices)** pour une gestion de mémoire efficace.
*   **Pipeline complet** : Préparation des données, ingénierie des fonctionnalités, entraînement de modèles multiples.
*   **Optimisation des hyperparamètres** via validation croisée.
*   **Sélection intelligente** des modèles selon la taille et la complexité du dataset.

## 📂 Architecture du Projet

```text
.
├── automl/                 # Cœur du package
│   ├── core.py             # Logique principale (pipeline AutoML)
│   ├── models.py           # Définitions des modèles et hyperparamètres
│   ├── preprocessing.py    # Nettoyage et préparation des données
│   ├── metrics.py          # Fonctions d'évaluation
│   └── utils.py            # Utilitaires (chargement de données, logs)
├── data_A/                 # Dossier contenant un dataset d'exemple
├── test_automl.py          # Script principal pour lancer l'analyse
├── requirements.txt        # Dépendances du projet
└── README.md               # Documentation
```

## 🚀 Installation

1.  **Prérequis** : Python 3.8 ou supérieur.
2.  Clonez ce dépôt ou téléchargez les fichiers.
3.  Installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

Le script `test_automl.py` permet de lancer l'AutoML sur les données présentes dans le répertoire courant.

### Lancer une analyse complète

```bash
python test_automl.py
```

Par défaut, le script analyse tous les sous-dossiers valides (contenant des fichiers `.data` et `.solution`) dans le répertoire actuel.

### Lancer sur un dataset spécifique

Pour exécuter l'AutoML uniquement sur le dataset `data_A` (inclus) :

```bash
python test_automl.py data_A
```

### Mode Debug

Pour afficher des logs détaillés sur le processus d'entraînement :

```bash
python test_automl.py --debug
```

## 🛠 Exemple de code (Intégration)

Vous pouvez utiliser le package `automl` directement dans vos propres scripts Python :

```python
import automl
import os

# Définir le dossier contenant les datasets
data_folder = os.getcwd()

# Lancer l'entraînement
automl.fit(data_folder, dataset_name='data_A')

# Évaluer et afficher les résultats
automl.eval()
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
