
import pandas as pd

customers = pd.read_csv("data/raw/customers.csv", keep_default_na=False)
orders = pd.read_csv("data/raw/orders.csv", keep_default_na=False)
products = pd.read_csv("data/raw/products.csv", keep_default_na=False)

print("CUSTOMER COLUMNS:", customers.columns.tolist())
print("\nRAW CITY VALUES:")
print(sorted(customers["city"].astype(str).unique()))

print("\nRAW GENDER VALUES:")
print(sorted(customers["gender"].astype(str).unique()))

print("\nCUSTOMER SAMPLE:")
print(customers.head(5).to_string(index=False))

print("\nORDER COLUMNS:", orders.columns.tolist())
print("\nPRODUCT COLUMNS:", products.columns.tolist())
