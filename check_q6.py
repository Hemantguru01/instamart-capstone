
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

query = """
WITH customer_spend AS (
    SELECT
        c.city,
        c.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
),
ranked AS (
    SELECT
        city,
        customer_id,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS spend_rank
    FROM customer_spend
)
SELECT
    city,
    COUNT(*) AS customers,
    MAX(spend_rank) AS max_rank
FROM ranked
GROUP BY city
ORDER BY city;
"""

result = pd.read_sql_query(query, conn)
print(result.to_string(index=False))

conn.close()
