# Instamart Customer Intelligence & Support Copilot

An end-to-end data analytics and customer-support project built with Python, pandas, SQLite, SQL, and machine-learning tools. The project prepares Instamart data for customer intelligence, customer segmentation, support-ticket triage, and policy-grounded support automation.

## Project Objectives

- Audit and clean raw customer, product, order, and ticket datasets.
- Store cleaned customer, product, and order data in SQLite.
- Analyze completed-order revenue using SQL and pandas.
- Develop customer intelligence and segmentation.
- Build a privacy-aware support-ticket triage workflow.
- Explore policy retrieval and tool-based support automation.

## Technology Stack

- Python 3
- pandas
- NumPy
- SQLite
- scikit-learn
- joblib
- SQL
- Git and GitHub
- VS Code

## Project Structure

```text
instamart-capstone/
├── data/
│   ├── raw/
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   └── tickets.csv
│   ├── policies/
│   └── instamart.db
├── sql/
│   └── queries.sql
├── state/
├── outputs/
├── generate_data.py
├── mock_llm.py
├── audit_raw.py
├── inspect_data.py
├── clean_customers.py
├── clean_orders.py
├── clean_products.py
├── clean_tickets.py
├── store_data.py
├── run_sql_queries.py
├── check_q6.py
├── validate_sql.py
├── pandas_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset Overview

| Dataset | Raw rows | Cleaned rows |
|---|---:|---:|
| Customers | 1,200 | 1,200 |
| Products | 40 | 40 |
| Orders | 13,084 | 12,309 |
| Tickets | 164 | 160 |

The raw data is retained separately from the cleaned data.

## Data Cleaning

### Customers

- Standardized city names.
- Normalized gender values to `female`, `male`, or `unknown`.
- Converted ages to numeric values.
- Handled missing and invalid ages using city-level median imputation.
- Standardized signup dates.

### Products

- Trimmed text fields.
- Converted prices to numeric values.
- Removed exact duplicate rows and duplicate product IDs.
- Excluded records with missing required fields or non-positive prices.

### Orders

- Removed exact duplicate rows.
- Deduplicated order IDs, retaining the latest timestamp.
- Converted numeric fields.
- Reconstructed missing order amounts using product price, quantity, and discount where applicable.
- Removed negative delivery times.
- Imputed missing delivery times for completed orders using city-level medians.
- Preserved missing delivery times for cancelled orders.

### Tickets

- Trimmed text fields and standardized categories.
- Removed exact duplicate rows and duplicate ticket IDs.
- Excluded tickets with missing required identifiers, text, or categories.
- Four blank-text tickets were excluded, leaving 160 usable tickets.

## SQLite Database

The database is located at `data/instamart.db` and contains these tables:

- `customers`
- `products`
- `orders`

The current database contains 1,200 customers, 40 products, and 12,309 orders.

## SQL Analytics

The `sql/queries.sql` file contains seven analytical queries:

1. Filtering with `WHERE` and `HAVING`
2. Three-table join for revenue by city and category
3. Scalar subquery for products above the catalogue average price
4. `LEFT JOIN` for customers without completed orders
5. Common table expression (CTE) for customer spending
6. Ranking window functions for customer spend by city
7. Running daily completed-order revenue

## Verified Results

The following results have been checked locally:

- Cleaned customers: 1,200
- Cleaned products: 40
- Cleaned orders: 12,309
- Usable tickets: 160
- Completed orders: 11,110
- Cancelled orders: 1,199
- Completed-order revenue: INR 2,998,668.80
- Daily revenue rows: 183
- SQL and pandas revenue totals match.
- The SQL validation script prints `ALL CHECKPOINTS PASSED`.

### Pandas Analysis

The `pandas_analysis.py` script includes:

- A city-by-category revenue pivot table.
- Per-city total spend, average order amount, and completed-order counts.
- Order amount distribution statistics.
- Interquartile range (IQR) outlier analysis.
- Correlations among order amount, quantity, and discount percentage.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate the raw data:

```powershell
python generate_data.py
```

Run the cleaning scripts:

```powershell
python clean_customers.py
python clean_products.py
python clean_orders.py
python clean_tickets.py
```

Build the SQLite database:

```powershell
python store_data.py
```

Run the SQL queries and validation:

```powershell
python run_sql_queries.py
python validate_sql.py
```

Run the pandas analysis:

```powershell
python pandas_analysis.py
```

## Current Status

The raw-data audit, cleaning scripts, SQLite database, seven SQL queries, and initial pandas analytics have been created and tested locally.

Customer intelligence and machine-learning deliverables, support-ticket triage, and policy retrieval and agent workflows should be documented as completed only after their implementation and evaluation checks pass.

## Data Privacy

- Do not commit credentials, API keys, or secrets.
- Avoid publishing raw ticket text or customer phone numbers.
- Review `.gitignore` before pushing the repository.
- Confirm whether the assignment permits committing generated data and SQLite database files before publishing them.

## Reproducibility

Run the scripts from the project root. Keep the raw datasets unchanged and regenerate cleaned outputs through the corresponding scripts.

