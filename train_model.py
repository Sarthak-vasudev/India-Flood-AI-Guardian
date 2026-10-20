
import os, numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import joblib

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")

for _f in ["data/indian_district_master.csv", "districts.csv", "indian_district_master.csv"]:
    _p = os.path.join(BASE, _f)
    if os.path.exists(_p):
        DATA_PATH = _p
        break
else:
    raise FileNotFoundError("No CSV found. Put indian_district_master.csv in data/ folder.")

os.makedirs(MODELS_DIR, exist_ok=True)

# FIX 1: safe per-file deletion (no shutil.rmtree)
for fname in ["knn_model.pkl", "scaler.pkl", "features.pkl"]:
    fpath = os.path.join(MODELS_DIR, fname)
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
            print(f"🗑️  Removed {fname}")
        except PermissionError:
            print(f"⚠️  Cannot delete {fname} (file in use). Will overwrite.")

FEATURES = [
    "live_precip_mm",
    "live_wind_kmh",
    "live_soil_moisture",
    "live_snowmelt_mm",
    "live_discharge",
    "elevation_m",
    "slope_deg",
    "soil_drainage",
    "historical_floods",
    "river_proximity_km",
]
LABEL_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Extreme": 3}

df = pd.read_csv(DATA_PATH)

# FIX 2: exclude fake district
df = df[df["district"] != "Bhubaneswar2"]

df["y"] = df["risk_label"].map(LABEL_MAP)
static = ["elevation_m", "slope_deg", "soil_drainage", "historical_floods", "river_proximity_km"]
df = df.dropna(subset=static + ["y"])
print(f"Loaded {len(df)} districts\n{df['risk_label'].value_counts()}\n")

# FIX 3: Proxy live values calibrated to actual Open-Meteo 7-day output
# These match what weather_fetcher.py actually returns at runtime
pm = {"Low": 15,  "Moderate": 60,  "High": 180,  "Extreme": 380}   # 7-day sum mm
wm = {"Low": 12,  "Moderate": 25,  "High": 45,   "Extreme": 72}    # km/h
sm = {"Low": 0.08,"Moderate": 0.18,"High": 0.30, "Extreme": 0.42}  # m3/m3 mean 8h
sn = {"Low": 0,   "Moderate": 0,   "High": 1,    "Extreme": 4}     # 7-day sum mm
dm = {"Low": 30,  "Moderate": 200, "High": 800,  "Extreme": 3000}  # m3/s

df["live_precip_mm"]     = df["risk_label"].map(pm)
df["live_wind_kmh"]      = df["risk_label"].map(wm)
df["live_soil_moisture"] = df["risk_label"].map(sm)
df["live_snowmelt_mm"]   = df["risk_label"].map(sn)
df["live_discharge"]     = df["risk_label"].map(dm)

np.random.seed(42)
rows = []

# ── Synthetic data — calibrated to Open-Meteo real output ranges ──────

# Extreme: heavy monsoon / cyclone / Himalayan discharge
for _ in range(500):
    rows.append([
        np.random.uniform(200, 600),    # 7-day rain: heavy monsoon
        np.random.uniform(45, 95),      # strong storm winds
        np.random.uniform(0.32, 0.50),  # saturated soil
        np.random.uniform(0, 15),       # some snowmelt (Himalayan)
        np.random.uniform(1500, 6000),  # high river discharge
        np.random.uniform(1, 120),      # low elevation flood plains
        np.random.uniform(0.1, 1.5),    # low slope = flat = floods
        np.random.randint(1, 3),        # poor drainage
        np.random.randint(6, 12),       # many historical floods
        np.random.uniform(0.1, 5),      # very close to river
        3  # Extreme
    ])

# High
for _ in range(450):
    rows.append([
        np.random.uniform(80, 250),
        np.random.uniform(28, 55),
        np.random.uniform(0.22, 0.36),
        np.random.uniform(0, 5),
        np.random.uniform(400, 1800),
        np.random.uniform(30, 350),
        np.random.uniform(0.3, 3.5),
        np.random.randint(2, 4),
        np.random.randint(3, 7),
        np.random.uniform(2, 25),
        2  # High
    ])

# Moderate
for _ in range(400):
    rows.append([
        np.random.uniform(20, 100),
        np.random.uniform(12, 32),
        np.random.uniform(0.12, 0.26),
        np.random.uniform(0, 2),
        np.random.uniform(80, 500),
        np.random.uniform(80, 600),
        np.random.uniform(0.8, 6.0),
        np.random.randint(2, 5),
        np.random.randint(1, 4),
        np.random.uniform(10, 70),
        1  # Moderate
    ])

# Low
for _ in range(350):
    rows.append([
        np.random.uniform(0, 30),       # FIX: was 0-8, too narrow vs real API
        np.random.uniform(0, 18),
        np.random.uniform(0.0, 0.14),
        0.0,
        np.random.uniform(5, 100),
        np.random.uniform(250, 2500),
        np.random.uniform(2.0, 18.0),
        np.random.randint(4, 9),
        np.random.randint(0, 2),
        np.random.uniform(25, 100),
        0  # Low
    ])

synth = pd.DataFrame(rows, columns=FEATURES + ["y"])
real  = df[FEATURES + ["y"]]
all_  = pd.concat([real, synth], ignore_index=True)

CLASS_NAMES = ["Low", "Moderate", "High", "Extreme"]

X = all_[FEATURES].values
y = all_["y"].values.astype(int)

# ── Honest evaluation split ────────────────────────────────────────────
# Evaluating the model on data it was trained on (as before) overstates
# performance for a distance-weighted KNN. Hold out a stratified 20% test
# set that the model never sees during fitting, and report all metrics on
# that split. The final model saved to disk is refit on 100% of the data
# afterward (standard practice: evaluate on a held-out split, then use all
# available data for the production model).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

sc = StandardScaler()
Xs_train = sc.fit_transform(X_train)
Xs_test  = sc.transform(X_test)

knn_eval = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean")
knn_eval.fit(Xs_train, y_train)

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn_eval, Xs_train, y_train, cv=cv)
print(f"CV Accuracy (train fold): {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Samples: {len(X)} total (real:{len(real)} synth:{len(synth)}) "
      f"| train:{len(X_train)} test:{len(X_test)}\n")

y_pred = knn_eval.predict(Xs_test)

# ── Evaluation metrics ──────────────────────────────────────────────────
accuracy          = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
precision_macro   = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro      = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro          = f1_score(y_test, y_pred, average="macro", zero_division=0)
kappa             = cohen_kappa_score(y_test, y_pred)
mcc               = matthews_corrcoef(y_test, y_pred)

print("=" * 60)
print("EVALUATION METRICS (held-out test set)")
print("=" * 60)
print(f"1. Accuracy               : {accuracy:.4f}")
print(f"2. Balanced Accuracy      : {balanced_accuracy:.4f}")
print(f"3. Precision (macro)      : {precision_macro:.4f}")
print(f"4. Recall (macro)         : {recall_macro:.4f}")
print(f"5. F1 Score (macro)       : {f1_macro:.4f}")
print(f"6. Cohen's Kappa          : {kappa:.4f}")
print(f"7. MCC                    : {mcc:.4f}")
print("=" * 60)

print("\n8. Per-class Precision / Recall / F1:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ── Confusion matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
ax.set_title("Confusion Matrix — KNN Flood Risk Classifier")
plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"✅ Confusion matrix saved to {cm_path}")

# ── Refit final model on ALL data for production use ────────────────────
sc_final = StandardScaler()
Xs_final = sc_final.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean")
knn.fit(Xs_final, y)
sc = sc_final  # keep downstream save code unchanged

joblib.dump(knn,      os.path.join(MODELS_DIR, "knn_model.pkl"))
joblib.dump(sc,       os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODELS_DIR, "features.pkl"))
print(f"✅ Saved to {MODELS_DIR}")
print("Now run: python -m streamlit run app.py")