# Bar_Total.py
# YE Dec 2024 — Total (Domestic + International) trips & spend by state
# Produces one vertical grouped bar chart + one CSV.
# Uses matplotlib only (no seaborn).

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------- Output location --------
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUT_DIR / "YE2024_total_trips_spend_by_state.csv"
PNG_OUT = OUT_DIR / "YE2024_total_trips_spend_vertical.png"

# -------- Data (YE Dec 2024) --------
# States order
states = ["NSW","VIC","QLD","SA","WA","TAS","NT","ACT"]

# Domestic (overnight) – absolute trips & AUD spend
dom_trips = [37484546, 29140729, 26187462,  7594207, 10854483,  3192242,  1422005,  3097872]
dom_spend = [114558355000, 80795356000, 99258069000, 25930524000, 49532981000, 12403023000, 7384356000, 7540310000]

# International – absolute trips & AUD spend
intl_trips = [ 4239300,  3004000,  2595400,   534200,  1039100,   339000,   259400,   214700]
intl_spend = [12025600000, 9046000000, 6220700000, 1347900000, 2625300000,  551300000,  468200000,  626500000]

# Totals
tot_trips = [d+i for d,i in zip(dom_trips,  intl_trips)]
tot_spend = [d+i for d,i in zip(dom_spend,  intl_spend)]

# Build CSV-friendly table (also include millions/$b)
df = pd.DataFrame({
    "state": states,
    "trips_absolute": tot_trips,
    "trips_millions": np.array(tot_trips)/1_000_000,
    "spend_aud_absolute": tot_spend,
    "spend_aud_billions": np.array(tot_spend)/1_000_000_000,
})
df.to_csv(CSV_OUT, index=False)
print(f"Wrote {CSV_OUT}")

# -------- Chart (vertical grouped bars) --------
x = np.arange(len(states))
width = 0.38

trips_m = df["trips_millions"].values
spend_b = df["spend_aud_billions"].values

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width/2, trips_m, width, label="Trips (millions)")
bars2 = ax.bar(x + width/2, spend_b, width, label="Spend ($b AUD)")

ax.set_title("YE Dec 2024 — Total (Domestic Overnight + International)\nTrips & Spend by State/Territory", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(states, rotation=0)
ax.set_ylabel("Value")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right")

# Value labels
def _annotate(bars, values):
    for rect, val in zip(bars, values):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h,
                f"{val:,.2f}", ha="center", va="bottom", fontsize=9)

_annotate(bars1, trips_m)
_annotate(bars2, spend_b)

fig.tight_layout()
plt.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {PNG_OUT}")
