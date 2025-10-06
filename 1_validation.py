# 1_validation.py — FIXED unique-place totals + Restaurants/Cafes sector
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_PATH = Path("data")
MERGED_FILENAME = "combined_dataNT.xlsx"   # ensure this file exists in /data
INPUT_FILE = DATA_PATH / MERGED_FILENAME

COLS = [
    "Category","text","rating","publishedDate",
    "placeInfo/address","placeInfo/name","placeInfo/numberOfReviews",
    "placeInfo/rating",
    "placeInfo/ratingHistogram/count1","placeInfo/ratingHistogram/count2",
    "placeInfo/ratingHistogram/count3","placeInfo/ratingHistogram/count4",
    "placeInfo/ratingHistogram/count5"
]
HIST_COLS = [f"placeInfo/ratingHistogram/count{i}" for i in range(1,6)]
SECTORS = ["Hotels", "Tours", "Activities", "Restaurants/Cafes", "Other"]

# ---------- Helpers ----------
def keep_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in df.columns if c in COLS]].copy()
    for c in COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[COLS]

# single regex that captures restaurants/cafes + spelling variants
RE_RC = re.compile(
    r"(restaurants?\s*[/&-]\s*caf(?:e|és|es)?|restaurant\s*[/&-]\s*caf(?:e|és|es)?|"
    r"\b(restauran|resturant|resturants?|restaurant|eatery|cafe|cafes|café|cafés|food|dining|bar|bistro|canteen)\b)",
    re.IGNORECASE
)

def normalize_sector(cat) -> str:
    """
    Map messy categories to canonical sectors.
    Restaurants + Cafes (and synonyms/misspellings) -> 'Restaurants/Cafes'
    """
    if pd.isna(cat) or str(cat).strip() == "":
        return "Other"
    c = str(cat).strip().lower()

    if RE_RC.search(c): return "Restaurants/Cafes"
    if re.search(r"hotel|motel|resort|lodg|hostel|accommodation", c): return "Hotels"
    if re.search(r"tour|charter|cruise|safari|expedition|guide|operator", c): return "Tours"
    if re.search(r"activity|attraction|market|park|beach|gallery|trail|zoo|wildlife|aquarium|lookout|mall|shopping", c): return "Activities"
    return "Other"

def to_num(x): return pd.to_numeric(x, errors="coerce")
def savefig(p): plt.tight_layout(); plt.savefig(p, bbox_inches="tight"); plt.close()

# ---------- Load ----------
df_any = pd.read_excel(INPUT_FILE, sheet_name=None)
df = pd.concat(df_any.values(), ignore_index=True) if isinstance(df_any, dict) else df_any
df = keep_cols(df)

# ---------- Parse numerics ----------
for c in ["rating","placeInfo/rating","placeInfo/numberOfReviews"] + HIST_COLS:
    if c in df.columns:
        df[c] = to_num(df[c])

# ---------- Sector mapping ----------
df["Sector"] = df["Category"].apply(normalize_sector)
print("Sector counts after mapping:\n", df["Sector"].value_counts(dropna=False))

# ---------- Shares & quality index ----------
hist_sum = df[HIST_COLS].fillna(0).sum(axis=1).replace(0, np.nan)
for i in range(1,6):
    df[f"share_{i}"] = (df[f"placeInfo/ratingHistogram/count{i}"].fillna(0) / hist_sum)
df["quality_index"] = df["share_5"].fillna(0) - (df["share_1"].fillna(0) + df["share_2"].fillna(0))
df["n_reviews"] = df["placeInfo/numberOfReviews"].fillna(0)

# ---------- Output dir ----------
OUT = Path("output/validation"); OUT.mkdir(parents=True, exist_ok=True)

# Overall missingness
miss = df.isna().sum().sort_values(ascending=False)
miss.to_csv(OUT / "overall_missing.csv")
print("\nOverall missing (top 10):\n", miss.head(10))

# Overall rating distribution
if df["rating"].notna().any():
    plt.figure(); df["rating"].dropna().hist(bins=20)
    plt.title("Overall rating distribution"); plt.xlabel("rating"); plt.ylabel("count")
    savefig(OUT / "overall_rating_hist.png")

# Reviews over time
if "publishedDate" in df.columns:
    dates = pd.to_datetime(df["publishedDate"], errors="coerce", utc=True)
    by_m = dates.dt.to_period("M").value_counts().sort_index()
    if len(by_m) > 0:
        plt.figure()
        x = np.arange(len(by_m))
        plt.plot(x, by_m.values)
        plt.xticks(x, [str(p) for p in by_m.index], rotation=45, ha="right")
        plt.title("Overall reviews per month"); plt.ylabel("count")
        savefig(OUT / "overall_reviews_over_time.png")

# ---------- KPI logic (fix: collapse to unique places) ----------
def sector_kpis_unique_places(frame: pd.DataFrame) -> pd.Series:
    """
    Collapse to unique places to avoid double-counting 'numberOfReviews'.
    - rating: use mean per place, then average across places
    - n_reviews: take max per place (the canonical total reviews for that place)
    - shares/quality_index: average per place
    """
    if frame.empty:
        return pd.Series({
            "places": 0,
            "avg_rating": np.nan,
            "med_rating": np.nan,
            "total_reviews": 0,
            "mean_share_5": np.nan,
            "mean_share_1": np.nan,
            "quality_index": np.nan,
        })

    by_place = frame.groupby("placeInfo/name", dropna=False).agg({
        "rating": "mean",
        "n_reviews": "max",            # <-- key fix (unique-place total)
        "share_5": "mean",
        "share_1": "mean",
        "quality_index": "mean"
    })

    return pd.Series({
        "places": by_place.shape[0],
        "avg_rating": by_place["rating"].mean(),
        "med_rating": by_place["rating"].median(),
        "total_reviews": by_place["n_reviews"].sum(),   # sum of unique-place totals
        "mean_share_5": by_place["share_5"].mean(),
        "mean_share_1": by_place["share_1"].mean(),
        "quality_index": by_place["quality_index"].mean(),
    })

# Overall sector KPIs table
rows = []
for sec in SECTORS:
    sub = df[df["Sector"] == sec]
    s = sector_kpis_unique_places(sub)
    s["Sector"] = sec
    rows.append(s)

agg = pd.DataFrame(rows).set_index("Sector").reindex(SECTORS)
agg.to_csv(OUT / "sector_kpis.csv")
print("\nSector KPIs (unique-place totals):\n", agg)

# ---------- Per-sector outputs ----------
for sec in SECTORS:
    sdir = OUT / sec.replace("/", "_").lower()
    sdir.mkdir(parents=True, exist_ok=True)

    dfs = df[df["Sector"] == sec].copy()
    if dfs.empty:
        pd.DataFrame({"note": [f"No rows for sector {sec}"]}).to_csv(sdir / "kpis.csv", index=False)
        continue

    # raw rows + missingness
    dfs.to_csv(sdir / "raw_rows.csv", index=False)
    dfs.isna().sum().sort_values(ascending=False).to_csv(sdir / "missing.csv")

    # rating histogram
    if dfs["rating"].notna().any():
        plt.figure(); dfs["rating"].dropna().hist(bins=20)
        plt.title(f"{sec} — rating distribution"); plt.xlabel("rating"); plt.ylabel("count")
        savefig(sdir / "rating_hist.png")

    # per-sector KPI using unique-place logic
    kpi = sector_kpis_unique_places(dfs).to_frame().T
    kpi.to_csv(sdir / "kpis.csv", index=False)

print("\nValidation done →", OUT.resolve())
