# bigquery_clean_two_datasets_no_nulls.py

import os
from google.cloud import bigquery
import pandas as pd

# 1️⃣ Set GCP credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"

# 2️⃣ Initialize BigQuery client
project_id = "recommendation-system-475301"
client = bigquery.Client(project=project_id)

# 3️⃣ Function to clean table and replace NULLs
def clean_table(dataset_id: str, table_name: str, destination_table: str):
    # Get all columns and types
    columns_info = client.query(f"""
        SELECT column_name, data_type
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
    """).to_dataframe()

    select_exprs = []
    for _, row in columns_info.iterrows():
        col = row['column_name']
        dtype = row['data_type']

        # Flatten arrays safely
        if dtype.startswith('ARRAY'):
            select_exprs.append(
                f"ARRAY(SELECT TO_JSON_STRING(x) FROM UNNEST({col}) AS x WHERE x IS NOT NULL) AS {col}_flat"
            )
        # Trim strings and replace NULL/empty with default
        elif dtype in ('STRING', 'CHAR', 'TEXT'):
            select_exprs.append(f"COALESCE(NULLIF(TRIM({col}), ''), 'Unknown') AS {col}_clean")
        # Replace NULL numeric types with 0
        elif dtype in ('INT64', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC'):
            select_exprs.append(f"COALESCE({col}, 0) AS {col}")
        # Replace NULL boolean with FALSE
        elif dtype == 'BOOL':
            select_exprs.append(f"COALESCE({col}, FALSE) AS {col}")
        # Keep dates/timestamps as-is
        else:
            select_exprs.append(col)

    select_sql = ",\n  ".join(select_exprs)

    # Build final query
    query = f"""
    WITH cleaned AS (
        SELECT DISTINCT
          {select_sql}
        FROM `{project_id}.{dataset_id}.{table_name}`
    )
    SELECT * FROM cleaned
    """

    # Execute query and save cleaned table
    job_config = bigquery.QueryJobConfig(
        destination=destination_table,
        write_disposition="WRITE_TRUNCATE"
    )
    client.query(query, job_config=job_config).result()
    print(f"✅ Cleaned table saved: {destination_table}")

# 4️⃣ Clean both datasets
clean_table(
    dataset_id="books",
    table_name="goodreads_books_mystery_thriller_crime",
    destination_table=f"{project_id}.books.goodreads_books_mystery_thriller_crime_cleaned"
)

clean_table(
    dataset_id="books",
    table_name="goodreads_interactions_mystery_thriller_crime",
    destination_table=f"{project_id}.books.goodreads_interactions_mystery_thriller_crime_cleaned"
)

# 5️⃣ Optional: fetch small samples to Python for verification
df_books_sample = client.query(
    f"SELECT * FROM `{project_id}.books.goodreads_books_mystery_thriller_crime_cleaned` LIMIT 5"
).to_dataframe()

df_interactions_sample = client.query(
    f"SELECT * FROM `{project_id}.books.goodreads_interactions_mystery_thriller_crime_cleaned` LIMIT 5"
).to_dataframe()

print("Books sample:")
print(df_books_sample.head())
print("\nInteractions sample:")
print(df_interactions_sample.head())
