import os
import pandas as pd

data_dir = "data"

cols = ["Category", "text", "rating", "publishedDate", "placeInfo/address", "placeInfo/name", "placeInfo/numberOfReviews"
        , "placeInfo/rating", "placeInfo/ratingHistogram/count1", "placeInfo/ratingHistogram/count2", "placeInfo/ratingHistogram/count3"
        , "placeInfo/ratingHistogram/count4", "placeInfo/ratingHistogram/count5"]

dfs = []

for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    
    if file.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        continue

    # Columns needed
    df = df[[c for c in df.columns if c in cols]]

    # Replace null value = NaN
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Order the columns
    df = df[cols]
    
    dfs.append(df)
    print(f"Read {file}: {df.shape[0]} rows, {df.shape[1]} columns")

df = pd.concat(dfs, ignore_index=True)
print(df.info())

