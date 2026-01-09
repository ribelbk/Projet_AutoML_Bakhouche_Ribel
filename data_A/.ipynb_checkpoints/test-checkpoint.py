import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import numpy as np


# --- Début du programme ---
print(f"[INFO] Programme démarré à {time.strftime('%H:%M:%S')}")

# --- Chargement des données (100 premières lignes) ---
X = pd.read_csv("data_C.data", sep="\s+")
y = pd.read_csv("data_C.solution", sep="\s+")
print(y) 
print(f"[INFO] Données chargées : X={X.shape}, y={y.shape}")

# --- Supprimer les lignes avec NaN ---
data = pd.concat([X, y], axis=1).dropna()
X = data[X.columns]
y = data[y.columns]
print(f"[INFO] Données après dropna : X={X.shape}, y={y.shape}")

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"[INFO] Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"[INFO] Test shape : X={X_test.shape}, y={y_test.shape}")

# --- Multi-output classifier ---
forest = RandomForestClassifier(random_state=42)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=1)
print("[INFO] Entraînement du modèle...")
multi_target_forest.fit(X_train, y_train)
print("[INFO] Modèle entraîné !")

# --- Prédictions ---
print("[INFO] Prédiction sur l'ensemble de test...")
y_pred = multi_target_forest.predict(X_test)
print("[INFO] Prédictions terminées !")

# --- Évaluation ---
print("[INFO] Évaluation du modèle...")
for i, col in enumerate(y.columns):
    y_true = y_test.iloc[:, i].to_numpy()
    y_hat = y_pred[:, i]
    acc = accuracy_score(y_true, y_hat)
    f1 = f1_score(y_true, y_hat, average='weighted')  # ou 'macro'
    print(f"{col} -> Accuracy: {acc:.4f}, F1-score: {f1:.4f}")


print(f"[INFO] Programme terminé à {time.strftime('%H:%M:%S')}")

