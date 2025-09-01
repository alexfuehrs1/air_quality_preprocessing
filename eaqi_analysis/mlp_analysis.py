import numpy as np
import pandas as pd
import os, joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# =========================
# Pfade & Konfiguration
# =========================
BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
CSV       = os.path.join(BASE_DIR, "ml_data", "ml_ready_eaqi_no3.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots", "mlp")
TARGET = "eaqi_partial_class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Optional: Thread-Limit für lineare Algebra (stabilere Laufzeit auf CPU)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

# =========================
# Daten laden
# =========================
df = pd.read_csv(CSV, parse_dates=["ts"])
df = df.drop(columns=["ts"], errors="ignore")
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# (Optional) float32 spart RAM & ist für MLP ausreichend
X = X.astype(np.float32)

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

# sample_weight (für weighted-Training)
sw_train = y_train.map(class_weights).to_numpy()

# =========================
# CPU-freundliche MLP-Settings
# =========================
# Hinweise:
# - hidden_layer_sizes: moderat (schnell + genug Kapazität)
# - batch_size: größer -> bessere CPU-Auslastung
# - early_stopping: True -> stoppt automatisch bei Plateau
# - n_iter_no_change/tol: sensible Frühstopp-Parameter
# - learning_rate_init: kleiner hilft bei Stabilität
# - max_iter: Limit, Early Stopping greift meist früher
mlp_params = dict(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,                 # L2
    batch_size=1024,            # CPU-geeignet
    learning_rate_init=5e-4,    # stabil
    max_iter=120,               # reicht mit Early Stopping
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
    tol=1e-4,
    shuffle=True,
    random_state=RANDOM_STATE,
    verbose=False               # auf True setzen falls du Logs willst
)

# Pipelines (mit Scaling)
mlp_plain = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    MLPClassifier(**mlp_params)
)

mlp_weighted = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    MLPClassifier(**mlp_params)
)

# =========================
# Training
# =========================
mlp_plain.fit(X_train, y_train)
# sample_weight in der Pipeline: parametername = "<step>__sample_weight"
mlp_weighted.fit(X_train, y_train, mlpclassifier__sample_weight=sw_train)

# Modelle speichern (komprimiert)
joblib.dump(mlp_plain,    os.path.join(MODEL_DIR, "mlp_plain.pkl"), compress=3)
joblib.dump(mlp_weighted, os.path.join(MODEL_DIR, "mlp_weighted.pkl"), compress=3)
print(f"[OK] Modelle gespeichert in: {MODEL_DIR}")

# =========================
# Evaluation + Plots
# =========================
def eval_model(name, model, filename_prefix):
    y_pred = model.predict(X_test)
    print("\n", "="*30, f"{name}", "="*30)
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")

    # Confusion Matrix (normalized)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix (normalized) - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion Matrix gespeichert: {out_path}")

eval_model("MLP (ohne Weights)", mlp_plain, "mlp_plain")
eval_model("MLP (mit Weights)",  mlp_weighted, "mlp_weighted")

# =========================
# (Optional) Permutation Importance (CPU-schonend)
# =========================
# Tipp: n_repeats klein halten; nutzt alle CPU-Kerne (n_jobs=-1).
# Falls es dir zu langsam ist, setze n_repeats=2 oder kommentiere den Block aus.
try:
    result = permutation_importance(
        mlp_weighted, X_test, y_test,
        n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    topn = 15

    plt.figure(figsize=(8,6))
    sns.barplot(x=importances.head(topn), y=importances.head(topn).index, palette="viridis")
    plt.title(f"Top-{topn} Permutation Importances (MLP weighted)")
    plt.xlabel("Mean decrease in score")
    plt.ylabel("Feature")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "mlp_weighted_permutation_importances.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Permutation-Importances gespeichert: {out_path}")
except Exception as e:
    print("[INFO] Permutation Importance übersprungen:", e)

try:
    result = permutation_importance(
        mlp_plain, X_test, y_test,
        n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    topn = 15

    plt.figure(figsize=(8,6))
    sns.barplot(x=importances.head(topn), y=importances.head(topn).index, palette="viridis")
    plt.title(f"Top-{topn} Permutation Importances (MLP plain)")
    plt.xlabel("Mean decrease in score")
    plt.ylabel("Feature")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "mlp_plain_permutation_importances.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Permutation-Importances gespeichert: {out_path}")
except Exception as e:
    print("[INFO] Permutation Importance übersprungen:", e)
