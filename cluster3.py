# nov_drop_diagnosis.py
# Purpose: Investigate decline in review counts/ratings (Nov 2024 – Jan 2025),
#          and compare BOTH seasons (Wet vs Dry) overall and by sector/region/emotion/text.
# Run: python nov_drop_diagnosis.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

INPUT_FILE = Path("data/combined_dataNTLatest.xlsx")
OUTDIR = Path("output/nov_drop")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def month_to_season(m):
    if pd.isna(m): return pd.NA
    m = int(m)
    # NT seasons:
    #   Dry = May–Oct
    #   Wet = Nov–Apr
    return "Dry (May–Oct)" if 5 <= m <= 10 else "Wet (Nov–Apr)"

def clean_text_to_tokens(text):
    if not isinstance(text, str) or not text.strip():
        return []
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)           # remove URLs
    t = re.sub(r"[^a-z\s]", " ", t)                   # keep letters/spaces
    t = re.sub(r"\s+", " ", t).strip()
    toks = t.split()
    stop = set("""
        the a an and or in on at for to from of with without be is are was were been
        it this that these those i we you they he she my our your their them us
        very really just much many more most least could would should can may might
        not no yes ok okay get got make made do did done
        visit visited visiting trip travel tourist tourists nt
    """.split())
    toks = [w for w in toks if w not in stop and len(w) >= 3]
    return toks

def count_words(series):
    ctr = Counter()
    for s in series:
        ctr.update(clean_text_to_tokens(s))
    return ctr

def top_word_lift(ctr_a, ctr_b, k=100, alpha=1.0):
    """
    Smoothed lift of vocab A relative to B:
      lift = ( (count_a+α)/(sum_a+α*V) ) / ( (count_b+α)/(sum_b+α*V) )
    """
    vocab = set(ctr_a) | set(ctr_b)
    sum_a = sum(ctr_a.values())
    sum_b = sum(ctr_b.values())
    rows = []
    V = len(vocab) if vocab else 1
    for w in vocab:
        a = ctr_a.get(w, 0)
        b = ctr_b.get(w, 0)
        p_a = (a + alpha) / (sum_a + alpha * V)
        p_b = (b + alpha) / (sum_b + alpha * V)
        rows.append((w, a, b, p_a / max(p_b, 1e-12)))
    out = pd.DataFrame(rows, columns=["word","count_A","count_B","lift"]).sort_values("lift", ascending=False)
    return out.head(k)

def save_bar(series_or_df, title, ylabel, fname, ylim=None):
    fig = plt.figure(figsize=(11,6))
    ax = plt.gca()
    if isinstance(series_or_df, pd.Series):
        series_or_df.plot(kind="bar", ax=ax)
    else:
        series_or_df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_barh(series, title, xlabel, fname, xlim=None):
    fig = plt.figure(figsize=(11,6))
    ax = plt.gca()
    series.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(*xlim)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

def stacked_emotion(df_in, index_col, title, fname):
    if "core_emotion" not in df_in.columns or df_in["core_emotion"].isna().all():
        return
    emo = df_in.groupby([index_col, "core_emotion"]).size().reset_index(name="count")
    if emo.empty:
        return
    emo["share"] = emo["count"] / emo.groupby(index_col)["count"].transform("sum")
    pivot = emo.pivot(index=index_col, columns="core_emotion", values="share").fillna(0)
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    bottom = np.zeros(len(pivot.index))
    for col in pivot.columns:
        ax.bar(pivot.index.astype(str), pivot[col].values, bottom=bottom, label=col)
        bottom += pivot[col].values
    ax.set_ylim(0,1)
    ax.set_title(title)
    ax.set_ylabel("Share of reviews (0–1)")
    plt.xticks(rotation=20, ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Emotion")
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# Load & prepare
# -------------------------
df = pd.read_excel(INPUT_FILE)
df["publishedDate"] = pd.to_datetime(df["publishedDate"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

for c in ["Sector","Category","Region","core_emotion","text","placeInfo/name"]:
    if c not in df.columns: df[c] = pd.NA

df["Month"]    = df["publishedDate"].dt.to_period("M")
df["Month_ts"] = df["Month"].dt.to_timestamp()
df["MonthNum"] = df["publishedDate"].dt.month
df["Season"]   = df["MonthNum"].apply(month_to_season)

# Focus window and baselines
win_start = pd.Timestamp("2024-11-01")
win_end   = pd.Timestamp("2025-01-31")
window   = df[(df["publishedDate"] >= win_start) & (df["publishedDate"] <= win_end)].copy()
# previous 3 months as baseline: Aug–Oct 2024
base_start = pd.Timestamp("2024-08-01")
base_end   = pd.Timestamp("2024-10-31")
baseline = df[(df["publishedDate"] >= base_start) & (df["publishedDate"] <= base_end)].copy()

# -------------------------
# 1) Overall monthly trends (context)
# -------------------------
monthly = (df.groupby("Month_ts")
             .agg(review_rows=("rating","size"),
                  avg_rating=("rating","mean"))
             .reset_index())
monthly.to_csv(OUTDIR / "monthly_overall.csv", index=False)

# Charts
fig = plt.figure(figsize=(11,5))
ax1 = plt.gca()
ax1.plot(monthly["Month_ts"], monthly["review_rows"], marker="o")
ax1.set_title("Monthly Review Count — overall")
ax1.set_ylabel("Review rows")
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(OUTDIR / "01_monthly_review_count.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(11,5))
ax2 = plt.gca()
ax2.plot(monthly["Month_ts"], monthly["avg_rating"], marker="o", color="tab:green")
ax2.set_title("Monthly Average Rating — overall")
ax2.set_ylabel("Average rating (0–5)")
ax2.set_ylim(0,5)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(OUTDIR / "02_monthly_avg_rating.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -------------------------
# 2) Window vs baseline — volume & rating
# -------------------------
win_months = (window.groupby("Month_ts")
                    .agg(review_rows=("rating","size"),
                         avg_rating=("rating","mean"))
                    .reset_index())
win_months.to_csv(OUTDIR / "window_months.csv", index=False)

# Median monthly count context
med_count = monthly["review_rows"].median()
save_bar(win_months.set_index("Month_ts")["review_rows"],
         "Review count (Nov 2024 – Jan 2025) vs overall median",
         f"Review rows (median={med_count:.0f})",
         "03_window_review_count.png")

# Average rating: window vs baseline (3 months prior)
win_rating  = window["rating"].mean()
base_rating = baseline["rating"].mean()
delta = pd.Series({"Window (Nov–Jan)": win_rating, "Baseline (Aug–Oct)": base_rating}).rename("Average rating (0–5)")
save_bar(delta, "Average rating — Window vs Baseline", "Average rating (0–5)", "04_window_vs_baseline_avg_rating.png", ylim=(0,5))

# -------------------------
# 3) Season analysis — BOTH seasons (overall and by month)
# -------------------------
season_overall = (df.groupby("Season")
                    .agg(review_rows=("rating","size"),
                         avg_rating=("rating","mean"))
                    .reset_index())
season_overall.to_csv(OUTDIR / "season_overall.csv", index=False)

save_bar(season_overall.set_index("Season")["review_rows"],
         "Review count by Season (overall)", "Review rows",
         "05_season_review_count_overall.png")

save_bar(season_overall.set_index("Season")["avg_rating"],
         "Average rating by Season (overall)", "Average rating (0–5)",
         "06_season_avg_rating_overall.png", ylim=(0,5))

# Monthly split by Season (full series)
season_month = (df.groupby(["Season","Month_ts"])
                  .agg(review_rows=("rating","size"),
                       avg_rating=("rating","mean"))
                  .reset_index())
season_month.to_csv(OUTDIR / "season_monthly.csv", index=False)

# In-window Season comparison (Nov, Dec, Jan)
win_season_counts = (df[df["Month_ts"].isin(win_months["Month_ts"])]
                     .groupby(["Season","Month_ts"]).size().unstack(0).fillna(0))
save_bar(win_season_counts,
         "Review count by Season — Nov, Dec, Jan (side-by-side)",
         "Review rows",
         "07_counts_by_season_window.png")

win_season_rating = (df[df["Month_ts"].isin(win_months["Month_ts"])]
                     .groupby(["Season","Month_ts"])["rating"].mean().unstack(0))
save_bar(win_season_rating,
         "Average rating by Season — Nov, Dec, Jan (side-by-side)",
         "Average rating (0–5)",
         "08_avg_rating_by_season_window.png", ylim=(0,5))

# -------------------------
# 4) Sector & Region contributions (window vs baseline)
# -------------------------
# Sector
sec_win = (window.groupby("Category")
                 .agg(review_rows=("rating","size"),
                      avg_rating=("rating","mean"))
                 .sort_values("review_rows", ascending=False))
sec_base = (baseline.groupby("Category")
                   .agg(base_rows=("rating","size"),
                        base_avg=("rating","mean")))
sec_compare = sec_win.join(sec_base, how="outer").fillna(0)
sec_compare["rating_delta_win_minus_base"] = sec_compare["avg_rating"] - sec_compare["base_avg"]
sec_compare.to_csv(OUTDIR / "sector_compare_window_vs_base.csv")
save_barh(sec_compare["review_rows"].sort_values(),
          "Window review count by Sector (Nov–Jan)",
          "Review rows",
          "09_window_sector_counts.png")
save_barh(sec_compare["rating_delta_win_minus_base"].sort_values(),
          "Rating change by Sector (window − baseline)",
          "Δ rating (0–5)",
          "10_sector_rating_delta.png", xlim=(-1,1))

# Region
reg_win = (window.groupby("Region")
                 .agg(review_rows=("rating","size"),
                      avg_rating=("rating","mean"))
                 .sort_values("review_rows", ascending=False))
reg_base = (baseline.groupby("Region")
                   .agg(base_rows=("rating","size"),
                        base_avg=("rating","mean")))
reg_compare = reg_win.join(reg_base, how="outer").fillna(0)
reg_compare["rating_delta_win_minus_base"] = reg_compare["avg_rating"] - reg_compare["base_avg"]
reg_compare.to_csv(OUTDIR / "region_compare_window_vs_base.csv")
save_barh(reg_compare["review_rows"].sort_values(),
          "Window review count by Region (Nov–Jan)",
          "Review rows",
          "11_window_region_counts.png")
save_barh(reg_compare["rating_delta_win_minus_base"].sort_values(),
          "Rating change by Region (window − baseline)",
          "Δ rating (0–5)",
          "12_region_rating_delta.png", xlim=(-1,1))

# -------------------------
# 5) Emotions — BOTH seasons
# -------------------------
if "core_emotion" in df.columns and df["core_emotion"].notna().any():
    # season overall emotion shares
    stacked_emotion(df, "Season",
                    "Emotion distribution by Season (overall)",
                    "13_emotion_by_season_overall.png")

    # emotion change in Window vs Baseline (overall)
    emo_win = window["core_emotion"].astype(str).value_counts(normalize=True)
    emo_base = baseline["core_emotion"].astype(str).value_counts(normalize=True)
    emo_compare = pd.concat([emo_win.rename("window_share"),
                             emo_base.rename("baseline_share")], axis=1).fillna(0)
    emo_compare["delta_win_minus_base"] = emo_compare["window_share"] - emo_compare["baseline_share"]
    emo_compare.sort_values("delta_win_minus_base", ascending=True).to_csv(OUTDIR / "emotion_window_vs_baseline.csv")

    save_barh(emo_compare["delta_win_minus_base"].sort_values(),
              "Emotion share change (Window − Baseline)",
              "Δ share (− worse / + better)",
              "14_emotion_delta_window_vs_base.png", xlim=(-0.5,0.5))

# -------------------------
# 6) Season × Sector and Season × Region (BOTH seasons)
# -------------------------
# Season × Sector
season_sector = (df.groupby(["Season","Category"])
                   .agg(review_rows=("rating","size"),
                        avg_rating=("rating","mean"))
                   .reset_index())
season_sector.to_csv(OUTDIR / "season_sector_summary.csv", index=False)

# Pivot for charts
piv_sec_cnt = season_sector.pivot(index="Category", columns="Season", values="review_rows").fillna(0)
piv_sec_rat = season_sector.pivot(index="Category", columns="Season", values="avg_rating").fillna(0)

save_bar(piv_sec_cnt,
         "Review count by Sector × Season (overall)",
         "Review rows",
         "15_sector_by_season_counts.png")

save_bar(piv_sec_rat.clip(0,5),
         "Average rating by Sector × Season (overall)",
         "Average rating (0–5)",
         "16_sector_by_season_avg_rating.png", ylim=(0,5))

# Season × Region
season_region = (df.groupby(["Season","Region"])
                   .agg(review_rows=("rating","size"),
                        avg_rating=("rating","mean"))
                   .reset_index())
season_region.to_csv(OUTDIR / "season_region_summary.csv", index=False)

piv_reg_cnt = season_region.pivot(index="Region", columns="Season", values="review_rows").fillna(0)
piv_reg_rat = season_region.pivot(index="Region", columns="Season", values="avg_rating").fillna(0)

save_bar(piv_reg_cnt,
         "Review count by Region × Season (overall)",
         "Review rows",
         "17_region_by_season_counts.png")

save_bar(piv_reg_rat.clip(0,5),
         "Average rating by Region × Season (overall)",
         "Average rating (0–5)",
         "18_region_by_season_avg_rating.png", ylim=(0,5))

# -------------------------
# 7) Text signals — Wet vs Dry comparisons
# -------------------------
wet  = df[df["Season"]=="Wet (Nov–Apr)"]
dry  = df[df["Season"]=="Dry (May–Oct)"]

wet_words = count_words(wet["text"])
dry_words = count_words(dry["text"])

lift_wet_over_dry = top_word_lift(wet_words, dry_words, k=100, alpha=1.0)
lift_dry_over_wet = top_word_lift(dry_words, wet_words, k=100, alpha=1.0)

lift_wet_over_dry.to_csv(OUTDIR / "words_overrepresented_in_wet_vs_dry.csv", index=False)
lift_dry_over_wet.to_csv(OUTDIR / "words_overrepresented_in_dry_vs_wet.csv", index=False)

# Quick bars (top 25 by lift)
if not lift_wet_over_dry.empty:
    save_bar(lift_wet_over_dry.head(25).set_index("word")["lift"],
             "Top words: over-represented in WET vs DRY",
             "Lift (↑ = more distinctive in Wet)",
             "19_words_lift_wet_over_dry.png")

if not lift_dry_over_wet.empty:
    save_bar(lift_dry_over_wet.head(25).set_index("word")["lift"],
             "Top words: over-represented in DRY vs WET",
             "Lift (↑ = more distinctive in Dry)",
             "20_words_lift_dry_over_wet.png")

print("✔️ Investigation saved to:", OUTDIR.resolve())
print("CSV tables:")
print(" - monthly_overall.csv")
print(" - window_months.csv")
print(" - season_overall.csv")
print(" - season_monthly.csv")
print(" - sector_compare_window_vs_base.csv")
print(" - region_compare_window_vs_base.csv")
print(" - emotion_window_vs_baseline.csv (if emotions exist)")
print(" - season_sector_summary.csv")
print(" - season_region_summary.csv")
print(" - words_overrepresented_in_wet_vs_dry.csv")
print(" - words_overrepresented_in_dry_vs_wet.csv")
print("Charts:")
print(" - 01_monthly_review_count.png")
print(" - 02_monthly_avg_rating.png")
print(" - 03_window_review_count.png")
print(" - 04_window_vs_baseline_avg_rating.png")
print(" - 05_season_review_count_overall.png")
print(" - 06_season_avg_rating_overall.png")
print(" - 07_counts_by_season_window.png")
print(" - 08_avg_rating_by_season_window.png")
print(" - 09_window_sector_counts.png")
print(" - 10_sector_rating_delta.png")
print(" - 11_window_region_counts.png")
print(" - 12_region_rating_delta.png")
print(" - 13_emotion_by_season_overall.png")
print(" - 14_emotion_delta_window_vs_base.png")
print(" - 15_sector_by_season_counts.png")
print(" - 16_sector_by_season_avg_rating.png")
print(" - 17_region_by_season_counts.png")
print(" - 18_region_by_season_avg_rating.png")
print(" - 19_words_lift_wet_over_dry.png")
print(" - 20_words_lift_dry_over_wet.png")
