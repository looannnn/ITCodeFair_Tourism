# sector_descriptive.py
# Run: python sector_descriptive.py

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = Path("data/combined_dataNTLatest.xlsx")  # your latest combined file
OUTDIR = Path("output/sectors")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Columns & helpers
# -----------------------------
COLS = [
    "Category", "text", "rating", "publishedDate",
    "placeInfo/address", "placeInfo/name", "placeInfo/numberOfReviews",
    "placeInfo/rating",
    "placeInfo/ratingHistogram/count1", "placeInfo/ratingHistogram/count2",
    "placeInfo/ratingHistogram/count3", "placeInfo/ratingHistogram/count4",
    "placeInfo/ratingHistogram/count5",
    # optional enrichments if present
    "core_emotion", "top_emotion", "confidence", "core_confidence",
    "Region", "city", "Month"
]

SECTOR_KEYWORDS = {
    "Hotels": ["hotel", "resort", "motel", "lodge", "lodg", "hostel", "accommodation"],
    "Activities": ["activity", "park", "beach", "zoo", "wildlife", "aquarium",
                   "museum", "gallery", "trail", "lookout", "safari", "expedition",
                   "tour", "charter", "cruise"],
    "Restaurants/Cafes": ["restaurant", "resturant", "cafe", "cafes", "food",
                          "dining", "bar", "pub", "eatery", "bistro", "bakery", "coffee"],
    "Tourist Attractions": ["attraction", "landmark", "monument", "heritage",
                            "site", "garden", "reserve", "botanic", "botanical"],
    "Shopping Malls": ["mall", "shopping", "plaza", "centre", "center"],
    "Night Markets": ["night market", "sunset market", "bazaar", "mindil"],
}

NUM_COLS = [
    "rating", "placeInfo/rating", "placeInfo/numberOfReviews",
    "placeInfo/ratingHistogram/count1", "placeInfo/ratingHistogram/count2",
    "placeInfo/ratingHistogram/count3", "placeInfo/ratingHistogram/count4",
    "placeInfo/ratingHistogram/count5",
]

SECTOR_ORDER = ["Hotels","Activities","Restaurants/Cafes","Tourist Attractions","Shopping Malls","Night Markets"]
SEASON_ORDER = ["Dry (May–Oct)", "Wet (Nov–Apr)"]

# -----------------------------
# Utility
# -----------------------------
def read_excel_all_sheets(path: Path) -> pd.DataFrame:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
        frames = []
        for sh, df in sheets.items():
            d = df.copy()
            d["__sheet"] = sh
            frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.read_excel(path)
    except Exception:
        return pd.read_excel(path)

def keep_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in df.columns if c in COLS]].copy()
    for c in COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[COLS]

def to_number(s):
    return pd.to_numeric(s, errors="coerce")

def map_sector(cat: str, name: str) -> str:
    if not pd.isna(cat):
        c = str(cat).lower()
        for sector, kws in SECTOR_KEYWORDS.items():
            if any(kw in c for kw in kws):
                return sector
    n = str(name).lower()
    for sector, kws in SECTOR_KEYWORDS.items():
        if any(kw in n for kw in kws):
            return sector
    # default to Tourist Attractions if unknown
    return "Tourist Attractions"

def month_to_season(m):
    if pd.isna(m): return pd.NA
    m = int(m)
    return "Dry (May–Oct)" if 5 <= m <= 10 else "Wet (Nov–Apr)"

# Plot helpers
def _fmt_thousands(x, pos):
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if x >= 1_000:     return f"{x/1_000:.1f}k"
    return f"{int(x)}"

def legend_outside(ax, title=None, fontsize=8):
    return ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                     fontsize=fontsize, frameon=False, title=title)

def barh_scaled(series, title, xlabel, fname, xlim=None, as_percent=False, log_scale=False):
    s = series.dropna().sort_values(ascending=True)
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.barh(s.index.astype(str), s.values)
    if as_percent:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.0%}"))
    elif xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    if log_scale:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

def stacked_bar_percent(df_pivot, title, ylabel, fname):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    bottom = np.zeros(len(df_pivot.index))
    for col in df_pivot.columns:
        ax.bar(df_pivot.index, df_pivot[col].values, bottom=bottom, label=str(col))
        bottom += df_pivot[col].values
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    legend_outside(ax, title="Legend", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

def scatter_scaled(x, y, labels, title, xlabel, ylabel, fname, xlim=None, ylim=None, add_45=False, logx=False):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(x, y)
    for i, txt in enumerate(labels):
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            ax.annotate(str(txt), (x[i], y[i]), xytext=(4, 4), textcoords='offset points', fontsize=8)
    if add_45:
        ax.plot([0, 5], [0, 5], "--", linewidth=1)
    if logx:
        ax.set_xscale("log")
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Load & clean
# -----------------------------
raw = read_excel_all_sheets(INPUT_FILE)
df = keep_cols(raw)

for c in NUM_COLS:
    if c in df.columns:
        df[c] = to_number(df[c])

df["Sector"] = df.apply(lambda r: map_sector(r.get("Category",""), r.get("placeInfo/name","")), axis=1)
df["review_row"] = 1  # each row is one review
df["rating"] = to_number(df["rating"])

# Month/Season
if "Month" in df.columns and df["Month"].notna().any():
    month_num = pd.to_datetime(df["Month"], errors="coerce").dt.month
else:
    month_num = pd.to_datetime(df["publishedDate"], errors="coerce").dt.month
df["Season"] = month_num.apply(month_to_season)

# Platform totals (deduped per place)
df["reported_reviews_place"] = (
    df.groupby("placeInfo/name")["placeInfo/numberOfReviews"]
      .transform("max").fillna(0).astype(float)
)

# Star shares & quality index
hist_cols = [f"placeInfo/ratingHistogram/count{i}" for i in range(1,6)]
if any(c in df.columns for c in hist_cols):
    hist_sum = df[hist_cols].fillna(0).sum(axis=1).replace(0, np.nan)
    for i in range(1,6):
        col = f"placeInfo/ratingHistogram/count{i}"
        df[f"share_{i}"] = (df[col].fillna(0)/hist_sum) if col in df.columns else np.nan
else:
    for i in range(1,6):
        df[f"share_{i}"] = np.nan

df["quality_index"] = df["share_5"].fillna(0) - (df["share_1"].fillna(0) + df["share_2"].fillna(0))

# -----------------------------
# Sector aggregates (accurate)
# -----------------------------
places_per_sector = df.groupby("Sector")["placeInfo/name"].nunique().rename("places")
reported_per_sector = (
    df.groupby(["Sector","placeInfo/name"])["reported_reviews_place"].max()
      .groupby("Sector").sum().rename("reported_reviews_total")
)
rows_per_sector = df.groupby("Sector")["review_row"].sum().rename("review_rows_dataset")

platform_reviews_per_place = (reported_per_sector / places_per_sector).rename("platform_reviews_per_place")
dataset_rows_per_place = (rows_per_sector / places_per_sector).rename("dataset_rows_per_place")
coverage_share = (rows_per_sector / reported_per_sector.replace(0, np.nan)).rename("dataset_share_of_platform")

rating_stats = df.groupby("Sector").agg(
    avg_rating=("rating","mean"),
    median_rating=("rating","median"),
    mean_share_5=("share_5","mean"),
    quality_index=("quality_index","mean"),
)

agg = pd.concat(
    [places_per_sector, rows_per_sector, reported_per_sector,
     platform_reviews_per_place, dataset_rows_per_place, coverage_share, rating_stats],
    axis=1
).reset_index()

# Leaderboards
rank_by_rating = agg.sort_values(["avg_rating","quality_index","median_rating"], ascending=[False,False,False])
rank_by_dataset_pop = agg.sort_values("review_rows_dataset", ascending=False)
rank_by_platform_pop = agg.sort_values("reported_reviews_total", ascending=False)

# -----------------------------
# Save tables (CSVs + Excel)
# -----------------------------
agg.to_csv(OUTDIR / "01_sector_summary.csv", index=False)
rank_by_rating.to_csv(OUTDIR / "02_sector_rank_by_rating.csv", index=False)
rank_by_dataset_pop.to_csv(OUTDIR / "03_sector_rank_by_dataset_rows.csv", index=False)
rank_by_platform_pop.to_csv(OUTDIR / "04_sector_rank_by_platform_totals.csv", index=False)
rank_by_rating.to_csv(OUTDIR / "Sector_rank_by_quality.csv", index=False)  # alias

with pd.ExcelWriter(OUTDIR / "sector_outputs.xlsx", engine="openpyxl") as w:
    df.to_excel(w, index=False, sheet_name="clean_rows")
    agg.to_excel(w, index=False, sheet_name="sector_summary")
    rank_by_rating.to_excel(w, index=False, sheet_name="rank_by_rating")
    rank_by_dataset_pop.to_excel(w, index=False, sheet_name="rank_by_pop_dataset")
    rank_by_platform_pop.to_excel(w, index=False, sheet_name="rank_by_pop_platform")

# -----------------------------
# Core Charts (ratings, popularity, leaderboards)
# -----------------------------
barh_scaled(agg.set_index("Sector")["avg_rating"], "Average rating by sector",
            "Average rating (0–5)", "Avg_rating_by_sector.png", xlim=(0,5))

barh_scaled(agg.set_index("Sector")["median_rating"], "Median rating by sector",
            "Median rating (0–5)", "Median_rating_by_sector.png", xlim=(0,5))

barh_scaled(agg.set_index("Sector")["mean_share_5"].fillna(0), "Mean 5★ share by sector",
            "Share of reviews (0–100%)", "Mean_p5_share_sector.png", as_percent=True)

barh_scaled(agg.set_index("Sector")["quality_index"].fillna(0),
            "Quality Index by sector (5★ − (1★+2★))", "Quality index (−1 to +1)",
            "Quality_index_by_sector.png", xlim=(-1,1))

barh_scaled(agg.set_index("Sector")["review_rows_dataset"],
            "Dataset review rows by sector (each row = one review)",
            "Review rows (dataset)", "Review_rows_by_sector.png")

barh_scaled(agg.set_index("Sector")["reported_reviews_total"],
            "Platform reported totals by sector (sum of per-place max, deduped)",
            "Platform reviews (log scale for readability)", "Reported_reviews_by_sector.png", log_scale=True)

barh_scaled(agg.set_index("Sector")["dataset_rows_per_place"],
            "Dataset rows per place (by sector)", "Rows per place (dataset)",
            "Rows_per_place_dataset_by_sector.png")

barh_scaled(agg.set_index("Sector")["platform_reviews_per_place"],
            "Platform reviews per place (by sector)", "Reviews per place (platform)",
            "Reviews_per_place_platform_by_sector.png")

barh_scaled(agg.set_index("Sector")["dataset_share_of_platform"].fillna(0),
            "Dataset coverage vs platform (rows ÷ platform totals)",
            "Coverage (0–100%)", "Dataset_coverage_share_by_sector.png", as_percent=True)

barh_scaled(rank_by_rating.set_index("Sector")["avg_rating"],
            "Ratings leaderboard (highest → lowest)", "Average rating (0–5)",
            "Ratings_leaderboard_by_sector.png", xlim=(0,5))

scatter_scaled(
    x=agg["review_rows_dataset"].values, y=agg["avg_rating"].values, labels=agg["Sector"].values,
    title="Average rating vs dataset review rows (by sector)",
    xlabel="Review rows (dataset)", ylabel="Average rating (0–5)",
    fname="Avg_rating_vs_review_rows.png", ylim=(0,5)
)
scatter_scaled(
    x=agg["reported_reviews_total"].values, y=agg["avg_rating"].values, labels=agg["Sector"].values,
    title="Average rating vs platform reported reviews (deduped, by sector)",
    xlabel="Reported reviews (platform)", ylabel="Average rating (0–5)",
    fname="Avg_rating_vs_reported_reviews.png", ylim=(0,5), logx=True
)

# -----------------------------
# Extras — distributions, correlations, places histograms
# -----------------------------
# Ratings distribution by sector (boxplot)
fig = plt.figure(figsize=(10,6))
order = agg.sort_values("avg_rating", ascending=True)["Sector"].tolist()
data = [df.loc[df["Sector"]==s, "rating"].dropna().values for s in order]
plt.boxplot(data, vert=False, labels=order, showfliers=False)
plt.xlim(0,5)
plt.title("Ratings distribution by sector (boxplot, no outliers)")
plt.xlabel("Rating (0–5)")
plt.tight_layout()
fig.savefig(OUTDIR / "Boxplot_ratings_by_sector.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Correlation heatmap — rating vs star shares
stars = ["share_1","share_2","share_3","share_4","share_5"]
for s in stars:
    if s not in df.columns:
        df[s] = np.nan
star_corr = df.groupby("Sector")[stars + ["rating"]].mean().corr()
fig = plt.figure(figsize=(6,5))
ax = plt.gca()
im = ax.imshow(star_corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(star_corr.shape[1])); ax.set_xticklabels(star_corr.columns, rotation=45, ha="right")
ax.set_yticks(range(star_corr.shape[0])); ax.set_yticklabels(star_corr.index)
plt.title("Correlation heatmap — rating vs star shares")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
fig.savefig(OUTDIR / "Heatmap_ratings_vs_starshares.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Reviews-per-place distributions
place = (df.groupby(["Sector","placeInfo/name"])
           .agg(dataset_rows=("review_row","sum"),
                platform_total=("reported_reviews_place","max"))
           .reset_index())

for col, title, fname in [
    ("dataset_rows",  "Histogram — dataset rows per place (by sector)",  "Histogram_dataset_rows_per_place.png"),
    ("platform_total","Histogram — platform reviews per place (by sector)","Histogram_platform_reviews_per_place.png"),
]:
    fig, ax = plt.subplots(figsize=(10,6))
    for s in agg["Sector"].unique():
        vals = place.loc[place["Sector"]==s, col].dropna().values
        if len(vals)==0: continue
        ax.hist(vals, bins=30, alpha=0.4, label=s)
    ax.set_title(title)
    ax.set_xlabel("Count"); ax.set_ylabel("Places")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    legend_outside(ax, title="Sectors", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Emotions (if present)
# -----------------------------
if "core_emotion" in df.columns:
    emo = (
        df.assign(core_emotion=df["core_emotion"].fillna("unknown"))
          .groupby(["Sector","core_emotion"]).size().reset_index(name="count")
    )
    emo["share"] = emo["count"] / emo.groupby("Sector")["count"].transform("sum")

    emo_pivot = emo.pivot(index="Sector", columns="core_emotion", values="share").fillna(0)
    stacked_bar_percent(
        emo_pivot, "Emotion distribution by sector", "Share of reviews (0–100%)",
        "Emotion_distribution_by_sector.png"
    )

    # Joy vs rating
    joy = emo[emo["core_emotion"].str.lower().eq("joy")].set_index("Sector")["share"]
    merged = agg.set_index("Sector").join(joy.rename("joy_share"))
    scatter_scaled(
        x=merged["avg_rating"].values, y=(merged["joy_share"].fillna(0)).values,
        labels=merged.index.values,
        title="Emotion vs Rating — Joy share by sector",
        xlabel="Average rating (0–5)", ylabel="Joy share (0–1)",
        fname="Emotion_vs_rating.png", xlim=(0,5), ylim=(0,1)
    )

    # Emotion radar per sector
    cats = sorted(emo["core_emotion"].dropna().str.lower().unique().tolist())
    for sector in emo["Sector"].unique():
        sub = emo[emo["Sector"]==sector].copy()
        shares = sub.groupby("core_emotion")["share"].sum()
        vals = [shares.get(c, 0.0) for c in cats]
        vals += vals[:1]
        angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylim(0, 1)
        plt.title(f"Emotion profile — {sector}")
        plt.tight_layout()
        fig.savefig(OUTDIR / f"Radar_emotions_{sector.replace('/','-')}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# -----------------------------
# NEW: Season analysis (Dry vs Wet) — sector + emotions
# -----------------------------
# Sector × Season summary
sector_season = df.groupby(["Sector","Season"]).agg(
    review_rows=("review_row","sum"),
    avg_rating=("rating","mean"),
    venues=("placeInfo/name","nunique"),
).reset_index()

sector_season.to_csv(OUTDIR / "Sector_season_summary.csv", index=False)

# 1) Review count by Sector × Season (grouped bars)
fig = plt.figure(figsize=(11,5))
ax = plt.gca()
width = 0.38
x = np.arange(len(SECTOR_ORDER))

dry = sector_season.pivot(index="Sector", columns="Season", values="review_rows").reindex(SECTOR_ORDER).fillna(0)
vals_dry = dry.get("Dry (May–Oct)", pd.Series(0, index=SECTOR_ORDER)).values
vals_wet = dry.get("Wet (Nov–Apr)", pd.Series(0, index=SECTOR_ORDER)).values

ax.bar(x - width/2, vals_dry, width, label="Dry (May–Oct)")
ax.bar(x + width/2, vals_wet, width, label="Wet (Nov–Apr)")
ax.set_xticks(x); ax.set_xticklabels(SECTOR_ORDER, rotation=20, ha="right")
ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
ax.set_title("Review count by Sector × Season")
ax.set_ylabel("Review rows (each row = 1 review)")
legend_outside(ax, "Season")
plt.tight_layout()
fig.savefig(OUTDIR / "S1_review_count_by_sector_season.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 2) Average rating by Sector × Season (grouped bars)
fig = plt.figure(figsize=(11,5))
ax = plt.gca()
dry_r = sector_season.pivot(index="Sector", columns="Season", values="avg_rating").reindex(SECTOR_ORDER)
vals_dry_r = dry_r.get("Dry (May–Oct)", pd.Series(np.nan, index=SECTOR_ORDER)).values
vals_wet_r = dry_r.get("Wet (Nov–Apr)", pd.Series(np.nan, index=SECTOR_ORDER)).values

ax.bar(x - width/2, vals_dry_r, width, label="Dry (May–Oct)")
ax.bar(x + width/2, vals_wet_r, width, label="Wet (Nov–Apr)")
ax.set_xticks(x); ax.set_xticklabels(SECTOR_ORDER, rotation=20, ha="right")
ax.set_ylim(0,5)
ax.set_title("Average rating by Sector × Season")
ax.set_ylabel("Average rating (0–5)")
legend_outside(ax, "Season")
plt.tight_layout()
fig.savefig(OUTDIR / "S2_avg_rating_by_sector_season.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 3) Season × Emotion (overall share)
if "core_emotion" in df.columns:
    df["core_emotion"] = df["core_emotion"].fillna("unknown").astype(str)

    emo_season = (
        df.groupby(["Season","core_emotion"]).size().reset_index(name="count")
          .assign(share=lambda d: d["count"] / d.groupby("Season")["count"].transform("sum"))
    )
    emo_season.to_csv(OUTDIR / "Season_emotion_summary.csv", index=False)

    piv = emo_season.pivot(index="Season", columns="core_emotion", values="share").fillna(0).reindex(SEASON_ORDER)
    fig = plt.figure(figsize=(9,5))
    ax = plt.gca()
    bottom = np.zeros(len(piv.index))
    for col in piv.columns:
        ax.bar(piv.index, piv[col].values, bottom=bottom, label=str(col))
        bottom += piv[col].values
    ax.set_ylim(0,1)
    ax.set_title("Emotion distribution by Season")
    ax.set_ylabel("Share of reviews (0–100%)")
    plt.xticks(rotation=0)
    legend_outside(ax, "Emotion")
    plt.tight_layout()
    fig.savefig(OUTDIR / "S3_emotion_distribution_by_season.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) Sector × Season × Emotion (share)
    emo_sector_season = (
        df.groupby(["Sector","Season","core_emotion"]).size().reset_index(name="count")
          .assign(share=lambda d: d["count"] / d.groupby(["Sector","Season"])["count"].transform("sum"))
    )
    emo_sector_season.to_csv(OUTDIR / "Sector_season_emotion_summary.csv", index=False)

    for sector in SECTOR_ORDER:
        sub = emo_sector_season[emo_sector_season["Sector"]==sector].copy()
        if sub.empty: continue
        piv2 = sub.pivot(index="Season", columns="core_emotion", values="share").fillna(0).reindex(SEASON_ORDER)

        fig = plt.figure(figsize=(9,4))
        ax = plt.gca()
        bottom = np.zeros(len(piv2.index))
        for col in piv2.columns:
            ax.bar(piv2.index, piv2[col].values, bottom=bottom, label=str(col))
            bottom += piv2[col].values
        ax.set_ylim(0,1)
        ax.set_title(f"Emotion distribution by Season — {sector}")
        ax.set_ylabel("Share of reviews (0–100%)")
        plt.xticks(rotation=0)
        legend_outside(ax, "Emotion")
        plt.tight_layout()
        safe = sector.replace("/","-")
        fig.savefig(OUTDIR / f"S4_emotion_by_season_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# -----------------------------
# Quadrant analysis — size vs satisfaction (platform totals)
# -----------------------------
x = agg["reported_reviews_total"].values
y = agg["avg_rating"].values
labels = agg["Sector"].values
x_med = np.nanmedian(x); y_med = np.nanmedian(y)

fig = plt.figure(figsize=(8,6))
ax = plt.gca()
ax.scatter(x, y)
for i, lab in enumerate(labels):
    ax.annotate(lab, (x[i], y[i]), xytext=(4,4), textcoords="offset points", fontsize=8)
ax.axvline(x_med, linestyle="--", color="gray")
ax.axhline(y_med, linestyle="--", color="gray")
ax.set_xscale("log"); ax.set_ylim(0,5)
ax.set_xlabel("Platform reported reviews (log)"); ax.set_ylabel("Average rating (0–5)")
ax.set_title("Quadrant analysis — size vs satisfaction")
plt.tight_layout()
fig.savefig(OUTDIR / "Quadrant_analysis_ratings_vs_reviews.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Plain-English sector stories (JSON)
# -----------------------------
stories = {}
if "core_emotion" in df.columns:
    emo_share = (df.assign(core_emotion=df["core_emotion"].fillna("unknown"))
                   .groupby(["Sector","core_emotion"]).size()
                   .groupby(level=0).apply(lambda x: (x/x.sum()).to_dict()))
else:
    emo_share = {}

for _, r in agg.iterrows():
    s    = r["Sector"]
    avg  = r.get("avg_rating", np.nan)
    rows = r.get("review_rows_dataset", np.nan)
    plat = r.get("reported_reviews_total", np.nan)
    qidx = r.get("quality_index", np.nan)
    places = r.get("places", np.nan)
    sentiment_hint = ""
    if s in emo_share and len(emo_share[s])>0:
        top_emo = sorted(emo_share[s].items(), key=lambda kv: kv[1], reverse=True)[0][0]
        sentiment_hint = f" Dominant emotion: {top_emo}."
    stories[s] = (
        f"{s}: avg {avg:.2f}★ across ~{int(rows):,} dataset reviews and ≈{int(plat):,} platform reviews, "
        f"{int(places):,} places. Quality index {qidx:.2f}.{sentiment_hint}"
    )

with open(OUTDIR / "sector_story.json", "w", encoding="utf-8") as f:
    json.dump(stories, f, indent=2, ensure_ascii=False)

print("✔️ Descriptive + Season outputs saved to:", OUTDIR.resolve())
