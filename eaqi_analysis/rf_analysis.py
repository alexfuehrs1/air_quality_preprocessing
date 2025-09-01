import numpy as np
import pandas as pd
from pathlib import Path
import os, joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# =========================
# Pfade & Konfiguration
# =========================
CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_data", "ml_ready_eaqi_no3.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots"))
TARGET = "eaqi_partial_class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =========================
# Daten laden
# =========================
df = pd.read_csv(CSV, parse_dates=["ts"])
# 'ts' ist nicht Feature
if "ts" in df.columns:
    df = df.drop(columns=["ts"])

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# =========================
# Class Weights berechnen
# =========================
classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))
print("Class Weights:", class_weights)

# =========================
# Train/Test-Split (stratifiziert)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# =========================
# Modelle
# =========================
rf_plain = RandomForestClassifier(
    n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
)

rf_weighted = RandomForestClassifier(
    n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE,
    class_weight=class_weights  
)

# =========================
# Training
# =========================
rf_plain.fit(X_train, y_train)
rf_weighted.fit(X_train, y_train)

# Modelle speichern
joblib.dump(rf_plain, MODEL_DIR + "/rf_plain.pkl")
joblib.dump(rf_weighted, MODEL_DIR + "/rf_weighted.pkl")

def eval_model(name, model, filename_prefix):
    y_pred = model.predict(X_test)
    print("\n", "="*30, f"{name}", "="*30)
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")

    # --- Confusion Matrix Plot ---
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix (normalized) - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = PLOT_DIR + f"/rf/{filename_prefix}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion Matrix gespeichert: {out_path}")

    return model

# Auswertung
rf_plain = eval_model("RandomForest (ohne Weights)", rf_plain, "rf_plain")
rf_weighted = eval_model("RandomForest (mit Weights)", rf_weighted, "rf_weighted")

# =========================
# Feature Importance Plot
# =========================
imp = pd.Series(rf_plain.feature_importances_, index=X.columns).sort_values(ascending=False)
topn = 15

plt.figure(figsize=(8,6))
sns.barplot(x=imp.head(topn), y=imp.head(topn).index, palette="viridis")
plt.title(f"Top-{topn} Feature Importances (plain RF)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
out_path = PLOT_DIR + "/rf/rf_plain_feature_importances.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[OK] Feature-Importances gespeichert: {out_path}")

imp = pd.Series(rf_weighted.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=imp.head(topn), y=imp.head(topn).index, palette="viridis")
plt.title(f"Top-{topn} Feature Importances (Weighted RF)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
out_path = PLOT_DIR + "/rf/rf_weighted_feature_importances.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[OK] Feature-Importances gespeichert: {out_path}")
