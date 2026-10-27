
from pathlib import Path
import sqlite3
import pandas as pd

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "instamart.db"
SQL_FILE = BASE / "sql" / "queries.sql"

sql_text = SQL_FILE.read_text(encoding="utf-8")

queries = {}
current_name = None
current_lines = []

for line in sql_text.splitlines():
    stripped = line.strip()

    if stripped.startswith("-- Q") and ":" in stripped:
        if current_name and current_lines:
            queries[current_name] = "\n".join(current_lines).strip()

        current_name = stripped[3:].split(":")[0].strip()
        current_lines = []
    elif current_name:
        current_lines.append(line)

if current_name and current_lines:
    queries[current_name] = "\n".join(current_lines).strip()

with sqlite3.connect(DB) as conn:
    for name, query in queries.items():
        query = "\n".join(
            line for line in query.splitlines()
            if not line.strip().startswith("--")
        ).strip()

        if not query:
            continue

        result = pd.read_sql_query(query, conn)

        print(f"\n{name}: {len(result)} rows")
        print(result.head(5).to_string(index=False))

        if name == "Q2":
            print("Total revenue:", round(result["revenue"].sum(), 2))

        if name == "Q7" and not result.empty:
            print("Final cumulative revenue:",
                  result["cumulative_revenue"].iloc[-1])
