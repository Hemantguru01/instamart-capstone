
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")

def audit_file(name):
    path = RAW / name
    df = pd.read_csv(path, keep_default_na=False)

    print(f"\n{'=' * 60}")
    print(f"FILE: {name}")
    print(f"SHAPE: {df.shape}")
    print("\nDATA TYPES:")
    print(df.dtypes.to_string())
    print("\nMISSING / BLANK VALUES:")
    for col in df.columns:
        blanks = df[col].astype(str).str.strip().eq("").sum()
        print(f"  {col}: {blanks}")

    print("\nSAMPLE:")
    print(df.head(3).to_string(index=False))
    return df

customers = audit_file("customers.csv")
products = audit_file("products.csv")
orders = audit_file("orders.csv")
tickets = audit_file("tickets.csv")

print("\nORDER CHECKPOINTS")
print("Rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending status:", orders["status"].astype(str).str.lower().eq("pending").sum())
print("Distinct order_id:", orders["order_id"].nunique())

amount = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
print("Non-numeric order_amount:", amount.isna().sum())
print("Negative delivery_minutes:", (delivery < 0).sum())
print("Blank delivery_minutes:",
      orders["delivery_minutes"].astype(str).str.strip().eq("").sum())

print("\nCUSTOMER CHECKPOINTS")
age = pd.to_numeric(customers["age"], errors="coerce")
gender = customers["gender"].astype(str).str.strip()
city = customers["city"].astype(str).str.strip()

print("Blank ages:", customers["age"].astype(str).str.strip().eq("").sum())
print("Ages outside 10–100:", ((age < 10) | (age > 100)).sum())
print("Distinct raw city spellings:", city.nunique())
print("Distinct raw gender spellings:", gender.nunique())
print("Blank genders:", gender.eq("").sum())
