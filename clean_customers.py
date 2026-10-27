
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(
    RAW / "customers.csv",
    keep_default_na=False
)

# 1. Standardize city spelling and whitespace
customers["city"] = (
    customers["city"]
    .astype(str)
    .str.strip()
    .str.title()
)

# 2. Normalize gender values
gender_map = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "": "unknown",
    "unknown": "unknown",
}
customers["gender"] = (
    customers["gender"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

# 3. Convert age to numeric and mark impossible ages missing
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)
invalid_age = ~customers["age"].between(10, 100)
customers.loc[invalid_age, "age"] = float("nan")

# Fill missing/invalid ages with the median for that city
city_median = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median)

# 4. Convert signup date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Save cleaned customer data
customers.to_csv(CLEAN / "customers.csv", index=False)

# Validation
print("Rows:", len(customers))
print("Cities:", sorted(customers["city"].unique()))
print("Number of cities:", customers["city"].nunique())
print("Gender values:", sorted(customers["gender"].unique()))
print("Missing ages:", customers["age"].isna().sum())
print("Invalid ages:", (~customers["age"].between(10, 100)).sum())
print("Missing signup dates:", customers["signup_date"].isna().sum())
print("Saved:", CLEAN / "customers.csv")
