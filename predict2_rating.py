# predict_rating.py
# Run: python predict_rating.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = Path("data/combined_dataNTLatest.xlsx")
OUTDIR = Path("output/predict")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Columns & helpers
# -----------------------------
COLS = [
    "Category", "placeInfo/name", "rating",
    "placeInfo/numberOfReviews", "placeInfo/rating",
    "placeInfo/ratingHistogram/count1", "placeInfo/ratingHistogram/count2",
    "placeInfo/ratingHistogram/count3", "placeInfo/ratingHistogram/count4",
    "placeInfo/ratingHistogram/count5",
]

SECTOR_KEYWORDS = {
    "Hotels": ["hotel", "resort", "motel", "lodge", "lodg", "hostel", "accommodation"],
    "Activities": ["activity", "tour", "charter", "cruise", "safari", "expedition",
                   "park", "beach", "zoo", "wildlife", "aquarium", "museum",
                   "gallery", "trail", "lookout"],
    "Restaurants/Cafes": ["restaurant", "resturant", "cafe", "cafes", "food",
                          "dining", "bar", "pub", "eatery", "bistro", "bakery", "coffee"],
    "Tourist Attractions": ["attraction", "landmark", "monument", "heritage",
                            "site", "garden", "reserve"],
    "Shopping Malls": ["mall", "shopping", "plaza", "centre", "center"],
    "Night Markets": ["night market", "sunset market", "bazaar", "mindil"],
}

def map_sector(cat: str, name: str) -> str:
    """Map noisy Category/place name into one of the 6 sectors."""
    if not pd.isna(cat):
        c = str(cat).lower()
        for sector, kws in SECTOR_KEYWORDS.items():
            if any(kw in c for kw in kws):
                return sector
    n = str(name).lower()
    for sector, kws in SECTOR_KEYWORDS.items():
        if any(kw in n for kw in kws):
            return sector
    return "Tourist Attractions"

def savefig(fig, name):
    fig.savefig(OUTDIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Load & prep
# -----------------------------
df = pd.read_excel(INPUT_FILE)
df = df[[c for c in COLS if c in df.columns]].copy()

# Sector
df["Sector"] = df.apply(lambda r: map_sector(r.get("Category",""), r.get("placeInfo/name","")), axis=1)

# Numeric features (coerce)
num_cols_raw = [c for c in [
    "placeInfo/numberOfReviews", "placeInfo/rating",
    "placeInfo/ratingHistogram/count1", "placeInfo/ratingHistogram/count2",
    "placeInfo/ratingHistogram/count3", "placeInfo/ratingHistogram/count4",
    "placeInfo/ratingHistogram/count5",
] if c in df.columns]
for c in num_cols_raw:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Derived features: log reviews + star-share proportions
df["log_num_reviews"] = np.log1p(df.get("placeInfo/numberOfReviews", pd.Series(np.nan)))

hist_cols = [c for c in num_cols_raw if "ratingHistogram" in c]
if len(hist_cols) == 5:
    total = df[hist_cols].fillna(0).sum(axis=1).replace(0, np.nan)
    for i in range(1,6):
        col = f"placeInfo/ratingHistogram/count{i}"
        df[f"share_{i}"] = (df[col] / total).clip(0,1)
else:
    for i in range(1,6):
        df[f"share_{i}"] = np.nan

# Target
df = df.dropna(subset=["rating"]).copy()
y = df["rating"].astype(float)

# Final feature set
num_cols = ["log_num_reviews", "placeInfo/rating"] + [f"share_{i}" for i in range(1,6)]
num_cols = [c for c in num_cols if c in df.columns]
X = df[["Sector"] + num_cols].copy()

# -----------------------------
# Model with robust imputation
# -----------------------------
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

pre = ColumnTransformer(
    transformers=[
        ("cat", cat_pipe, ["Sector"]),
        ("num", num_pipe, num_cols),
    ],
    remainder="drop"
)

model = Pipeline([("prep", pre), ("reg", Ridge(alpha=1.0, random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
r2  = r2_score(y_test, pred_test)
mae = mean_absolute_error(y_test, pred_test)
print(f"Test R^2: {r2:.3f}")
print(f"Test MAE: {mae:.3f}")

cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print(f"CV R^2 (mean ± sd): {cv.mean():.3f} ± {cv.std():.3f}")

joblib.dump(model, OUTDIR / "model.joblib")
print("Saved model:", (OUTDIR / "model.joblib").resolve())

# -----------------------------
# Predict all rows & save
# -----------------------------
df["pred_rating"] = model.predict(X)
df["residual"] = df["rating"].astype(float) - df["pred_rating"]

out_cols = ["placeInfo/name","Sector","rating","pred_rating","residual"] + [c for c in num_cols if c in df.columns]
df[out_cols].to_csv(OUTDIR / "predictions_all_rows.csv", index=False)

sector_cmp = df.groupby("Sector").agg(
    actual_avg     = ("rating","mean"),
    predicted_avg  = ("pred_rating","mean"),
    count          = ("rating","size"),
    abs_error_mean = ("residual", lambda s: np.mean(np.abs(s))),
).reset_index()
sector_cmp["abs_gap"] = (sector_cmp["actual_avg"] - sector_cmp["predicted_avg"]).abs()
sector_cmp = sector_cmp.sort_values("actual_avg", ascending=False)
sector_cmp.to_csv(OUTDIR / "sector_pred_vs_actual.csv", index=False)

# -----------------------------
# Plots — clear, comparative, scaled
# -----------------------------
# 1) Predicted vs Actual (test) with 45° line
fig = plt.figure(figsize=(6,6))
plt.scatter(y_test, pred_test)
plt.plot([0,5], [0,5], "--", linewidth=1)
plt.xlim(0,5); plt.ylim(0,5)
plt.xlabel("Actual rating (test)")
plt.ylabel("Predicted rating (test)")
plt.title("Prediction accuracy — test set")
savefig(fig, "P1_pred_vs_actual_test_scatter.png")

# 2) Sector average — Actual vs Predicted
sec = sector_cmp.set_index("Sector")[["actual_avg","predicted_avg"]]
fig = plt.figure(figsize=(9,5))
sec.plot(kind="barh")
plt.xlim(0,5)
plt.title("Sector average rating — Actual vs Predicted")
plt.xlabel("Average rating (0–5)")
savefig(plt.gcf(), "P2_sector_average_actual_vs_predicted.png")

# 3) Mean Absolute Error by sector
fig = plt.figure(figsize=(8,5))
plt.barh(sector_cmp["Sector"], sector_cmp["abs_error_mean"])
plt.xlabel("Mean Absolute Error (MAE)")
plt.title("Prediction error by sector")
savefig(fig, "P3_sector_mae.png")

# 4) Residuals by sector (boxplot)
order = sec.index.tolist()
fig = plt.figure(figsize=(9,5))
data = [df.loc[df["Sector"]==s, "residual"].values for s in order]
plt.boxplot(data, vert=False, labels=order, showfliers=False)
plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Residuals by sector (actual − predicted)")
plt.xlabel("Residual")
savefig(fig, "P4_sector_residuals_boxplot.png")

# 5) Calibration curve (deciles of predicted)
df["pred_bin"] = pd.qcut(df["pred_rating"], q=10, duplicates="drop")
cal = df.groupby("pred_bin").agg(predicted=("pred_rating","mean"),
                                 actual=("rating","mean")).reset_index()
fig = plt.figure(figsize=(7,5))
plt.plot(cal["predicted"], cal["actual"], marker="o")
plt.plot([0,5],[0,5],"--")
plt.xlim(0,5); plt.ylim(0,5)
plt.title("Calibration — actual vs predicted by decile")
plt.xlabel("Predicted (mean)")
plt.ylabel("Actual (mean)")
savefig(fig, "P5_calibration_curve.png")

# 6) Error vs venue popularity (|residual| vs log reviews)
if "placeInfo/numberOfReviews" in df.columns:
    fig = plt.figure(figsize=(8,5))
    plt.scatter(np.log1p(df["placeInfo/numberOfReviews"].clip(lower=0)),
                np.abs(df["residual"]))
    plt.xlabel("log(1 + platform reviews)")
    plt.ylabel("Absolute error |residual|")
    plt.title("Error vs venue popularity")
    savefig(fig, "P6_error_vs_popularity.png")

# 7) Feature importance (Ridge coefficients): OHE(Sector) + numeric features
ohe = model.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
reg = model.named_steps["reg"]
sector_fn  = [f"Sector={s}" for s in ohe.get_feature_names_out(["Sector"])]
num_fn     = num_cols
feat_names = sector_fn + num_fn

coefs = reg.coef_.ravel()
imp = pd.Series(coefs, index=feat_names).sort_values()

top_k = 12
imp_top = pd.concat([imp.head(top_k), imp.tail(top_k)])

fig = plt.figure(figsize=(8,7))
imp_top.plot(kind="barh")
plt.title("Feature importance (Ridge coefficients)\nNegative lowers, positive raises predicted rating")
plt.xlabel("Coefficient")
savefig(fig, "P7_feature_importance.png")

print("✔️ Prediction outputs saved to:", OUTDIR.resolve())
