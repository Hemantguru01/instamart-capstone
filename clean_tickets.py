
from pathlib import Path
import pandas as pd

# Project paths
BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
CLEAN = BASE / "data" / "clean"

CLEAN.mkdir(parents=True, exist_ok=True)

# Load tickets
tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

print("Raw ticket rows:", len(tickets))

# Standardize text fields
for column in ["ticket_id", "customer_id", "order_id", "text", "true_category"]:
    tickets[column] = tickets[column].astype(str).str.strip()

# Remove exact duplicate rows
tickets = tickets.drop_duplicates()

# Keep one record per ticket ID
tickets = tickets.drop_duplicates(
    subset=["ticket_id"],
    keep="first"
)

# Validate required fields
tickets = tickets[
    (tickets["ticket_id"] != "")
    & (tickets["customer_id"] != "")
    & (tickets["text"] != "")
    & (tickets["true_category"] != "")
].copy()

# Standardize category spelling
tickets["true_category"] = tickets["true_category"].str.lower()

# Save cleaned tickets
output_path = CLEAN / "tickets.csv"
tickets.to_csv(output_path, index=False)

# Validation report
print("Cleaned ticket rows:", len(tickets))
print("Distinct ticket IDs:", tickets["ticket_id"].nunique())
print("Missing ticket IDs:", (tickets["ticket_id"] == "").sum())
print("Missing customer IDs:", (tickets["customer_id"] == "").sum())
print("Blank ticket texts:", (tickets["text"] == "").sum())
print("Categories:", sorted(tickets["true_category"].unique()))
print("Saved:", output_path)
