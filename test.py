import pandas as pd

# Load the filtered .parquet file
df = pd.read_parquet("output/100BT_au/000_00000_au.parquet")

au_count = df["url"].str.contains(r"\.au", case=False, na=False).sum()
print(f"{au_count} out of {len(df)} rows have .au in the URL")

# View first few rows
print(df.head())

# Optionally print URLs
print(df["url"].dropna().head(20))
