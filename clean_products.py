
from pathlib import Path
import pandas as pd

# Project paths
BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
CLEAN = BASE / "data" / "clean"

CLEAN.mkdir(parents=True, exist_ok=True)

# Load raw products
products = pd.read_csv(
    RAW / "products.csv",
    keep_default_na=False
)

print("Raw product rows:", len(products))

# Standardize text columns
for column in ["product_id", "product_name", "category"]:
    products[column] = products[column].astype(str).str.strip()

# Convert price to numeric
products["price_inr"] = pd.to_numeric(
    products["price_inr"],
    errors="coerce"
)

# Remove exact duplicate rows
products = products.drop_duplicates()

# Keep one record per product ID
products = products.drop_duplicates(
    subset=["product_id"],
    keep="first"
)

# Validate required fields
products = products[
    (products["product_id"] != "")
    & (products["product_name"] != "")
    & (products["category"] != "")
    & (products["price_inr"].notna())
    & (products["price_inr"] > 0)
].copy()

# Save cleaned products
output_path = CLEAN / "products.csv"
products.to_csv(output_path, index=False)

# Validation report
print("Cleaned product rows:", len(products))
print("Distinct product IDs:", products["product_id"].nunique())
print("Missing product IDs:", (products["product_id"] == "").sum())
print("Missing product names:", (products["product_name"] == "").sum())
print("Missing categories:", (products["category"] == "").sum())
print("Missing prices:", products["price_inr"].isna().sum())
print("Invalid prices:", (products["price_inr"] <= 0).sum())
print("Saved:", output_path)
