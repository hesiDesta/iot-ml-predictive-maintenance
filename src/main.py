"""Studi kasus IoT: Predictive Maintenance Motor Industri dengan Machine Learning.

Mengikuti alur kerja pada materi:
 1. Kumpulkan data  2. Preprocessing  3. Split train/test
 4. Pilih algoritma 5. Latih model    6. Evaluasi   7. Deploy (inference)
Mencakup supervised (klasifikasi & regresi) dan unsupervised (clustering & anomali).
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             mean_absolute_error, r2_score, silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from generate_data import generate, LABELS

SEED = 42
FITUR = ["suhu", "getaran", "arus", "rpm", "tekanan"]
for d in ("data", "results", "models"):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- 1. Data
path = "data/sensor_motor.csv"
if not os.path.exists(path):
    generate().to_csv(path, index=False)
df = pd.read_csv(path)
print(f"[1] Data dimuat: {df.shape}")

# ---------------------------------------------------------- 2. Preprocessing
mv = df.isna().sum()
print(f"[2] Missing value:\n{mv[mv > 0].to_string()}")
n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"    Duplikat dihapus: {n_dup} | ukuran sekarang: {df.shape}")
# Missing value diisi median di dalam pipeline (mencegah data leakage)

X = df[FITUR]
y = df["status"]

# ---------------------------------------------------------- 3. Split 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"[3] Train: {len(X_train)} | Test: {len(X_test)}")

# ------------------------------------- 4-6. SUPERVISED: Klasifikasi (label)
print("\n[4-6] KLASIFIKASI status motor (supervised)")
kandidat = {
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
}
hasil = {}
for nama, model in kandidat.items():
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    pipe.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, pipe.predict(X_train))
    acc_te = accuracy_score(y_test, pipe.predict(X_test))
    cv = cross_val_score(pipe, X_train, y_train, cv=5).mean()
    hasil[nama] = (pipe, acc_te)
    # Cek overfitting: akurasi train jauh di atas test = tanda overfitting
    print(f"  {nama:14s} train={acc_tr:.3f} test={acc_te:.3f} cv5={cv:.3f} "
          f"gap={acc_tr - acc_te:+.3f}")

terbaik = max(hasil, key=lambda k: hasil[k][1])
model_terbaik = hasil[terbaik][0]
pred = model_terbaik.predict(X_test)
print(f"\n  Model terbaik: {terbaik}")
print(classification_report(y_test, pred, target_names=list(LABELS.values())))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=list(LABELS.values())).plot(
    ax=ax, xticks_rotation=30, colorbar=False)
ax.set_title(f"Confusion Matrix - {terbaik}")
plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=130); plt.close()

est = model_terbaik.steps[-1][1]
if hasattr(est, "feature_importances_"):
    imp = pd.Series(est.feature_importances_, index=FITUR).sort_values()
    imp.plot.barh(title="Pentingnya fitur sensor"); plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=130); plt.close()

# ------------------------------------------- SUPERVISED: Regresi (kontinu)
print("[REGRESI] Prediksi suhu motor dari sensor lain (nilai kontinu)")
fr = ["getaran", "arus", "rpm", "tekanan"]
dr = df.dropna(subset=["suhu"])
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    dr[fr], dr["suhu"], test_size=0.2, random_state=SEED)
reg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    LinearRegression()).fit(Xr_tr, yr_tr)
pr = reg.predict(Xr_te)
print(f"  MAE={mean_absolute_error(yr_te, pr):.2f} C | R2={r2_score(yr_te, pr):.3f}")

# ----------------------------------------- UNSUPERVISED: Clustering K-Means
print("\n[CLUSTERING] K-Means pada data tanpa label")
Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    print(f"  k={k} silhouette={silhouette_score(Xs, km.labels_):.3f}")
km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Xs)
xy = PCA(n_components=2, random_state=SEED).fit_transform(Xs)  # reduksi dimensi
plt.figure(figsize=(6, 5))
plt.scatter(xy[:, 0], xy[:, 1], c=km.labels_, s=8, cmap="viridis")
plt.title("K-Means (k=4) divisualisasikan dengan PCA")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
plt.savefig("results/kmeans_pca.png", dpi=130); plt.close()

# ------------------------------------------ UNSUPERVISED: Deteksi anomali
print("\n[ANOMALI] Isolation Forest")
iso = IsolationForest(contamination=0.30, random_state=SEED).fit(Xs)
is_anom = iso.predict(Xs) == -1
nyata_rusak = (y != 0).values
print(f"  Terdeteksi anomali: {is_anom.sum()} | Kondisi rusak sebenarnya: {nyata_rusak.sum()}")
print(f"  Recall kerusakan: {(is_anom & nyata_rusak).sum() / nyata_rusak.sum():.3f}")

# ------------------------------------------------ 7. Deploy: inference baru
joblib.dump(model_terbaik, "models/model_status_motor.joblib")
model = joblib.load("models/model_status_motor.joblib")
baru = pd.DataFrame([
    {"suhu": 54.0, "getaran": 2.1, "arus": 10.2, "rpm": 1448, "tekanan": 5.0},
    {"suhu": 78.0, "getaran": 2.3, "arus": 11.5, "rpm": 1440, "tekanan": 5.1},
    {"suhu": 61.0, "getaran": 5.9, "arus": 10.1, "rpm": 1450, "tekanan": 5.0},
    {"suhu": 56.0, "getaran": 2.0, "arus": 14.8, "rpm": 1380, "tekanan": 6.3},
])
print("\n[7] INFERENCE data sensor baru")
for i, p in enumerate(model.predict(baru)):
    print(f"  Sampel {i + 1}: {LABELS[int(p)]}")
