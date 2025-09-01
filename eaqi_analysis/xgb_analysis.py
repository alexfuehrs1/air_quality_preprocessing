import numpy as np
import pandas as pd
import os, joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
# =========================
# Pfade & Konfiguration
# =========================
CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_data", "ml_ready_eaqi_no3.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots"))
os.makedirs(os.path.join(PLOT_DIR, "xgb"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "eaqi_partial_class"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15              # Anteil (relativ zu Train) für Early Stopping

# =========================
# Daten laden
# =========================
df = pd.read_csv(CSV, parse_dates=["ts"])
if "ts" in df.columns:
    df = df.drop(columns=["ts"])

y_raw = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# (robust): Labels 0..K-1 encoden – XGB kommt zwar auch mit {1,2,6} klar,
# aber so sind num_class & Reports konsistent.
le = LabelEncoder()
y = le.fit_transform(y_raw)          # z.B. {1,2,6} -> {0,1,2}
classes_enc = np.unique(y)
class_names = le.inverse_transform(classes_enc)  # für Achsenbeschriftung

# =========================
# Class Weights berechnen
# =========================
weights = compute_class_weight(class_weight="balanced", classes=classes_enc, y=y)
class_weights = dict(zip(classes_enc, weights))
print("Class Weights (encoded):", class_weights)

# =========================
# Train/Test-Split (stratifiziert)
# =========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Validation-Split für Early Stopping (aus dem Trainingsanteil)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE, stratify=y_train_full, random_state=RANDOM_STATE
)

# Sample Weights für das gewichtete Modell
sample_weight = np.vectorize(class_weights.get)(y_train)

# =========================
# Modelle (XGBoost)
# =========================
# Basis-Hyperparameter (bewährt für Tabulardaten; passe gern an)
base_params = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=len(classes_enc),
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="mlogloss",
    early_stopping_rounds=50,   # <- WICHTIG: hierher (in den ctor), nicht in fit()
)

xgb_plain = XGBClassifier(**base_params)
xgb_weighted = XGBClassifier(**base_params)

eval_set = [(X_val, y_val)]

# Plain
xgb_plain.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

# Weighted
xgb_weighted.fit(
    X_train, y_train,
    sample_weight=sample_weight,
    eval_set=eval_set,
    verbose=False
)

# Modelle speichern
joblib.dump(xgb_plain, os.path.join(MODEL_DIR, "xgb_plain.pkl"))
joblib.dump(xgb_weighted, os.path.join(MODEL_DIR, "xgb_weighted.pkl"))

# =========================
# Evaluations-Helfer
# =========================
def eval_model(name, model, filename_prefix):
    y_pred = model.predict(X_test)
    print("\n", "="*30, f"{name}", "="*30)
    # Report wieder in Originalklassen benennen
    print(classification_report(le.inverse_transform(y_test),
                                le.inverse_transform(y_pred),
                                digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=classes_enc)
    cm_norm = confusion_matrix(y_test, y_pred, labels=classes_enc, normalize="true")

    # --- Confusion Matrix Plot ---
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (normalized) - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "xgb", f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion Matrix gespeichert: {out_path}")

# Auswertung
eval_model("XGBoost (ohne Weights)", xgb_plain, "xgb_plain")
eval_model("XGBoost (mit Weights)", xgb_weighted, "xgb_weighted")

# =========================
# Feature Importances (Gain)
# =========================
def xgb_gain_importances(model, feature_names):
    """
    Liefert Gain-Importances als pd.Series, robust für:
    - Keys = 'f0','f1',... (kein Feature-Name im Booster)
    - Keys = echte Spaltennamen (z.B. 'eco2')
    """
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")  # dict

    # Falls leer (kann in seltenen Fällen passieren), gib 0en zurück
    if not score:
        return pd.Series(0.0, index=feature_names)

    keys = list(score.keys())

    # Fall A: 'f#'-Schlüssel -> auf Spalten mappen
    if all(k.startswith("f") and k[1:].isdigit() for k in keys):
        idx_to_name = {f"f{i}": col for i, col in enumerate(feature_names)}
        series = pd.Series({idx_to_name[k]: v for k, v in score.items() if k in idx_to_name})
        # Sicherstellen, dass alle Features vertreten sind
        series = series.reindex(feature_names).fillna(0.0)

    # Fall B: echte Feature-Namen -> direkt verwenden
    else:
        series = pd.Series(score)
        # Importe auf vollständige Featureliste bringen (fehlende = 0)
        series = series.reindex(feature_names).fillna(0.0)

    return series.sort_values(ascending=False)


def xgb_gain_importances(model, feature_names):
    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type='gain')
    s = pd.Series(gain_dict)
    
    # Prüfen: sind die Keys schon echte Spaltennamen?
    if all(k in feature_names for k in s.index):
        # ja: alles gut
        pass
    else:
        # nein: fallback auf Mapping f0->Feature
        mapping = {f"f{i}": name for i, name in enumerate(feature_names)}
        s.index = [mapping.get(k, k) for k in s.index]
    
    s = s.reindex(feature_names, fill_value=0.0)
    total = s.sum()
    if total > 0:
        s = s / total
    return s.sort_values(ascending=False)

imp = xgb_gain_importances(xgb_plain, X.columns)
topn = 15

plt.figure(figsize=(8,6))
sns.barplot(x=imp.head(topn), y=imp.head(topn).index, palette="viridis")
plt.title(f"Top-{topn} Feature Importances (Gain, normalisiert) – XGB (plain)")
plt.xlabel("Importance (Summe=1)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "xgb", "xgb_plain_feature_importances_norm.png"), dpi=150)
plt.close()

imp = xgb_gain_importances(xgb_weighted, X.columns)

plt.figure(figsize=(8,6))
sns.barplot(x=imp.head(topn), y=imp.head(topn).index, palette="viridis")
plt.title(f"Top-{topn} Feature Importances (Gain, normalisiert) – XGB (weighted)")
plt.xlabel("Importance (Summe=1)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "xgb", "xgb_weighted_feature_importances_norm.png"), dpi=150)
plt.close()
