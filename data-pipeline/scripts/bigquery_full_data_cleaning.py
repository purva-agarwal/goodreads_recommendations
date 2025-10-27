# bigquery_clean_books_and_interactions_fixed.py

import os
import logging
from google.cloud import bigquery

class DataCleaning:
    
    def __init__(self):
        # Set Google Application Credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"
        
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.median_numeric_cols = ["publication_year", "num_pages"]

        # BigQuery client
        self.client = bigquery.Client()
        self.project_id = self.client.project
        

    def clean_table(self, dataset_id: str, table_name: str, destination_table: str, apply_global_median: bool = False):

        try:
            self.logger.info(f"Starting cleaning for table: {dataset_id}.{table_name}")

            # Get all columns
            columns_info = self.client.query(f"""
                SELECT column_name, data_type
                FROM `{self.project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).to_dataframe(create_bqstorage_client=False)
            self.logger.info(f"Retrieved {len(columns_info)} columns for table {table_name}")

            array_cols = [row['column_name'] for _, row in columns_info.iterrows() if row['data_type'].startswith('ARRAY')]
            string_cols = [row['column_name'] for _, row in columns_info.iterrows() if row['data_type'] in ('STRING', 'CHAR', 'TEXT')]
            bool_cols = [row['column_name'] for _, row in columns_info.iterrows() if row['data_type'] == 'BOOL']

            # Build select expressions
            select_exprs = []
            for _, row in columns_info.iterrows():
                col = row['column_name']

                if apply_global_median and col in self.median_numeric_cols:
                    select_exprs.append(
                        f"COALESCE({col}, global_medians.{col}_median) AS {col}"
                    )
                elif col in string_cols:
                    select_exprs.append(f"COALESCE(NULLIF(TRIM({col}), ''), 'Unknown') AS {col}_clean")
                elif col in bool_cols:
                    select_exprs.append(f"COALESCE({col}, FALSE) AS {col}")
                elif col in array_cols:
                    select_exprs.append(
                        f"ARRAY(SELECT TO_JSON_STRING(x) FROM UNNEST({col}) AS x WHERE x IS NOT NULL) AS {col}_flat"
                    )
                else:
                    select_exprs.append(col)

            select_sql = ",\n  ".join(select_exprs)

            # Build query
            if apply_global_median:
                query = f"""
                WITH main AS (
                    SELECT *
                    FROM `{self.project_id}.{dataset_id}.{table_name}`
                ),
                global_medians AS (
                    SELECT
                        {', '.join([f'APPROX_QUANTILES({col}, 2)[OFFSET(1)] AS {col}_median' for col in self.median_numeric_cols])}
                    FROM main
                )
                SELECT DISTINCT
                {select_sql}
                FROM main
                LEFT JOIN global_medians ON TRUE
                """
            else:
                query = f"""
                SELECT DISTINCT
                {select_sql}
                FROM `{self.project_id}.{dataset_id}.{table_name}`
                """

            # Execute query and save to destination table
            self.logger.info(f"Executing cleaning query for {table_name}...")
            job_config = bigquery.QueryJobConfig(
                destination=destination_table,
                write_disposition="WRITE_TRUNCATE"
            )
            self.client.query(query, job_config=job_config).result()
            self.logger.info(f" Cleaned table saved: {destination_table}")

        except Exception as e:
            self.logger.error(f"Error cleaning table {dataset_id}.{table_name}: {e}", exc_info=True)

    def main_runner(self):
        # Clean books and interactions tables

        self.clean_table(
            dataset_id="books",
            table_name="goodreads_books_mystery_thriller_crime",
            destination_table=f"{self.project_id}.books.goodreads_books_cleaned",
            apply_global_median=True
        )

        self.clean_table(
            dataset_id="books",
            table_name="goodreads_interactions_mystery_thriller_crime",
            destination_table=f"{self.project_id}.books.goodreads_interactions_cleaned",
            apply_global_median=False
        )

        # Fetch and log sample rows from cleaned tables
        try:
            df_books_sample = self.client.query(
                f"SELECT * FROM `{self.project_id}.books.goodreads_books_cleaned` LIMIT 5"
            ).to_dataframe(create_bqstorage_client=False)

            df_interactions_sample = self.client.query(
                f"SELECT * FROM `{self.project_id}.books.goodreads_interactions_cleaned` LIMIT 5"
            ).to_dataframe(create_bqstorage_client=False)

            self.logger.info("Books sample:")
            self.logger.info("\n%s", df_books_sample)
            self.logger.info("Interactions sample:")
            self.logger.info("\n%s", df_interactions_sample)
        except Exception as e:
            self.logger.error(f"Error fetching sample data: {e}", exc_info=True)
            print("Books sample:")

def main():
    # data_cleaner = DataCleaning()
    # data_cleaner.main_runner()
    print("TEST COMMENT")

if __name__ == "__main__":
    main()
