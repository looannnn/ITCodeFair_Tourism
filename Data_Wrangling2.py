import os
import pandas as pd

# =========================
# Config
# =========================
data_dir = "data"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

cols = [
    "Category", "text", "rating", "publishedDate",
    "placeInfo/address", "placeInfo/name", "placeInfo/numberOfReviews",
    "placeInfo/rating",
    "placeInfo/ratingHistogram/count1", "placeInfo/ratingHistogram/count2",
    "placeInfo/ratingHistogram/count3", "placeInfo/ratingHistogram/count4",
    "placeInfo/ratingHistogram/count5"
]

dfs = []

def standardize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in[[c for c in df_in.columns if c in cols]].copy()
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]

# =========================
# Read all files
# =========================
for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)

    if file.endswith(".csv"):
        df_raw = pd.read_csv(file_path)
        df_std = standardize_columns(df_raw)
        dfs.append(df_std)
        print(f"Read {file}: {df_std.shape[0]} rows")

    elif file.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(file_path, sheet_name=None)  # dict of sheetname->df
        for sheet_name, df_sheet in sheets.items():
            df_std = standardize_columns(df_sheet)
            dfs.append(df_std)
            print(f"Read {file} [{sheet_name}]: {df_std.shape[0]} rows")

# =========================
# Combine and save
# =========================
if dfs:
    df = pd.concat(dfs, ignore_index=True)

    print("\n=== Combined DataFrame ===")
    print(df.info())
    print(df.head(5))

    # Save CSV
    csv_path = os.path.join(output_dir, "combined.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV file: {csv_path}")

    # Save Excel with openpyxl (no extra install needed)
    xlsx_path = os.path.join(output_dir, "combined.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="combined")
    print(f"Saved Excel file: {xlsx_path}")

else:
    print("No data files found in folder:", data_dir)
