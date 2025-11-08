"""
Data Cleaning Module for Goodreads Recommendation System

This module handles the cleaning and preprocessing of raw Goodreads data from BigQuery.
It performs data validation, null handling, text cleaning, and creates author gender mappings.

Key Features:
- Cleans books and interactions tables from BigQuery
- Handles missing values with appropriate defaults
- Standardizes text fields and removes duplicates
- Creates author gender mapping using gender-guesser library
- Applies author median imputation for publication year anf number of pages
- Global outlier detection

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datapipeline.scripts.logger_setup import get_logger
import time
from datetime import datetime, timedelta
from gender_guesser.detector import Detector
from tqdm import tqdm
import sys

class DataCleaning:
    
    def __init__(self):
        """
        Initialize the DataCleaning class with BigQuery client and configuration.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration for data cleaning operations
        - BigQuery client and project information
        - Column names for median imputation
        """
        # Set Google Application Credentials for BigQuery access
        # Uses AIRFLOW_HOME environment variable to locate credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"
        
        # Initialize logging for data cleaning operations
        self.logger = get_logger("data_cleaning")
        self.author_median_cols = ["publication_year", "num_pages"]
        self.client = bigquery.Client()
        self.project_id = self.client.project
        

    def clean_table(self, dataset_id: str, table_name: str, destination_table: str, apply_global_median: bool = False):
        try:
            print(f"\n=== Starting cleaning for table: {dataset_id}.{table_name} ===")
            print("Destination table:", destination_table)

            # Get table schema
            columns_info = self.client.query(f"""
                SELECT column_name, data_type
                FROM `{self.project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).to_dataframe(create_bqstorage_client=False)
            print(f"Retrieved {len(columns_info)} columns for table {table_name}")

            array_cols = [r['column_name'] for _, r in columns_info.iterrows() if r['data_type'].startswith('ARRAY')]
            string_cols = [r['column_name'] for _, r in columns_info.iterrows() if r['data_type'] in ('STRING', 'CHAR', 'TEXT')]
            bool_cols = [r['column_name'] for _, r in columns_info.iterrows() if r['data_type'] == 'BOOL']
            print("Columns categorized:")
            print("Array columns:", array_cols)
            print("String columns:", string_cols)
            print("Boolean columns:", bool_cols)

            # Build column cleaning expressions
            select_exprs = []
            for _, row in columns_info.iterrows():
                col = row['column_name']
                if col in string_cols:
                    select_exprs.append(f"COALESCE(NULLIF(TRIM({col}), ''), 'Unknown') AS {col}_clean")
                elif col in bool_cols:
                    select_exprs.append(f"COALESCE({col}, FALSE) AS {col}")
                elif col in array_cols:
                    select_exprs.append(f"ARRAY(SELECT TO_JSON_STRING(x) FROM UNNEST({col}) AS x WHERE x IS NOT NULL) AS {col}_flat")
                else:
                    select_exprs.append(col)

            select_sql = ",\n  ".join(select_exprs)
            print("Select expressions built for all columns.")

            if apply_global_median:
                print("Applying author-wise median and outlier detection for numeric columns:", self.author_median_cols)
                query = f"""
                WITH main AS (
                SELECT b.*, a.author_id
                FROM `{self.project_id}.{dataset_id}.{table_name}` b,
                UNNEST(b.authors) AS a
                ),

                -- Step 1: Compute author-wise medians (for cleaning)
                author_medians AS (
                    SELECT
                        author_id,
                        APPROX_QUANTILES(NULLIF(publication_year, 0), 2)[OFFSET(1)] AS publication_year_median,
                        APPROX_QUANTILES(NULLIF(num_pages, 0), 2)[OFFSET(1)] AS num_pages_median
                    FROM main
                    GROUP BY author_id
                ),

                -- Step 2: Compute global medians (fallback)
                global_medians AS (
                    SELECT
                        APPROX_QUANTILES(NULLIF(publication_year, 0), 2)[OFFSET(1)] AS global_publication_year_median,
                        APPROX_QUANTILES(NULLIF(num_pages, 0), 2)[OFFSET(1)] AS global_num_pages_median
                    FROM main
                ),

                -- Step 3: Replace missing/zero values using author median, else global median
                cleaned AS (
                    SELECT
                        m.*,
                        CASE 
                            WHEN m.publication_year IS NULL OR m.publication_year = 0 THEN 
                                COALESCE(a.publication_year_median, g.global_publication_year_median)
                            ELSE m.publication_year
                        END AS publication_year_cleaned,

                        CASE 
                            WHEN m.num_pages IS NULL OR m.num_pages = 0 THEN 
                                COALESCE(a.num_pages_median, g.global_num_pages_median)
                            ELSE m.num_pages
                        END AS num_pages_cleaned
                    FROM main m
                    LEFT JOIN author_medians a ON m.author_id = a.author_id
                    CROSS JOIN global_medians g
                ),

                -- Step 4: Compute global quartiles on cleaned values
                global_stats AS (
                    SELECT
                        APPROX_QUANTILES(publication_year_cleaned, 4)[OFFSET(1)] AS publication_year_q1,
                        APPROX_QUANTILES(publication_year_cleaned, 4)[OFFSET(3)] AS publication_year_q3,
                        APPROX_QUANTILES(num_pages_cleaned, 4)[OFFSET(1)] AS num_pages_q1,
                        APPROX_QUANTILES(num_pages_cleaned, 4)[OFFSET(3)] AS num_pages_q3
                    FROM cleaned
                )

                -- Step 5: Flag outliers using global IQR rule
                SELECT DISTINCT
                    {select_sql},
                    c.publication_year_cleaned,
                    c.num_pages_cleaned,
                    CASE
                        WHEN c.publication_year_cleaned < (g.publication_year_q1 - 1.5 * (g.publication_year_q3 - g.publication_year_q1))
                        OR c.publication_year_cleaned > (g.publication_year_q3 + 1.5 * (g.publication_year_q3 - g.publication_year_q1))
                        OR c.num_pages_cleaned < (g.num_pages_q1 - 1.5 * (g.num_pages_q3 - g.num_pages_q1))
                        OR c.num_pages_cleaned > (g.num_pages_q3 + 1.5 * (g.num_pages_q3 - g.num_pages_q1))
                        THEN TRUE ELSE FALSE
                    END AS is_outlier
                FROM cleaned c
                CROSS JOIN global_stats g;

                """
            else:
                print("Cleaning without median/outlier detection.")
                query = f"""
                SELECT DISTINCT
                    {select_sql},
                    FALSE AS is_outlier
                FROM `{self.project_id}.{dataset_id}.{table_name}`
                """

            print("SQL query constructed. Submitting BigQuery job...")
            job_config = bigquery.QueryJobConfig(
                destination=destination_table,
                write_disposition="WRITE_TRUNCATE"
            )
            self.client.query(query, job_config=job_config).result()
            print(f"BigQuery job completed successfully. Cleaned table saved: {destination_table}")

        except Exception as e:
            print(f"❌ Error cleaning table {dataset_id}.{table_name}: {e}")
            import traceback
            traceback.print_exc()


    def run(self):
        """
        Execute the complete data cleaning pipeline.

        - Books table: applies author-wise median imputation for publication_year and num_pages,
        plus outlier detection based on 1.5*IQR rule per author.
        - Interactions table: simple cleaning, no median or outlier detection.
        """
        self.logger.info("=" * 60)
        self.logger.info("Good Reads Data Cleaning Pipeline")
        start_time = time.time()
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # === Clean books table ===
        self.clean_table(
            dataset_id="books",
            table_name="goodreads_books_mystery_thriller_crime",
            destination_table=f"{self.project_id}.books.goodreads_books_cleaned_staging_new",
            apply_global_median=True  # Author-wise median + outlier detection
        )

        # === Clean interactions table ===
        self.clean_table(
            dataset_id="books",
            table_name="goodreads_interactions_mystery_thriller_crime",
            destination_table=f"{self.project_id}.books.goodreads_interactions_cleaned_staging_new",
            apply_global_median=False  # No median, no outlier
        )

        # Fetch and log sample rows for verification
        try:
            df_books_sample = self.client.query(
                f"SELECT * FROM `{self.project_id}.books.goodreads_books_cleaned_staging_new` LIMIT 5"
            ).to_dataframe(create_bqstorage_client=False)
            df_interactions_sample = self.client.query(
                f"SELECT * FROM `{self.project_id}.books.goodreads_interactions_cleaned_staging_new` LIMIT 5"
            ).to_dataframe(create_bqstorage_client=False)
            self.logger.info("Books sample:\n%s", df_books_sample)
            self.logger.info("Interactions sample:\n%s", df_interactions_sample)
        except Exception as e:
            self.logger.error(f"Error fetching sample data: {e}", exc_info=True)
            print("Books sample:")
            
        # Create author gender mapping for bias analysis
        self.create_author_gender_map()
        
        # Log pipeline completion statistics
        end_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
        self.logger.info("=" * 60)

    def create_author_gender_map(self):
        """
        Generate and upload author gender mapping table to BigQuery.
        
        This method creates a gender mapping for authors to support bias analysis
        in the recommendation system. It uses the gender-guesser library to infer
        gender from author names and stores the results in BigQuery.
        
        The gender mapping is used later in the bias analysis pipeline to ensure
        fair recommendations across different author demographics.
        """
        try:
            self.logger.info("Starting gender mapping for authors...")

            # Load authors table from BigQuery using dynamic project ID
            query = f"""
                SELECT author_id, name
                FROM `{self.project_id}.books.goodreads_book_authors`
                WHERE name IS NOT NULL
            """
            authors_df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            self.logger.info(f"Retrieved {len(authors_df)} author rows.")

            # Initialize gender detector with case-insensitive matching
            detector = Detector(case_sensitive=False)

            def get_gender(name):
                """
                Infer gender from author name using gender-guesser library.
                
                Args:
                    name (str): Author's full name
                    
                Returns:
                    str: 'Male', 'Female', or 'Unknown'
                """
                # Handle edge cases: empty names, names with periods, or single characters
                if not name or '.' in name or len(name.split()) == 0:
                    return "Unknown"
                    
                # Use first name for gender inference
                g = detector.get_gender(name.split()[0])
                
                # Map gender-guesser results to our categories
                if g in ["male", "mostly_male"]:
                    return "Male"
                elif g in ["female", "mostly_female"]:
                    return "Female"
                else:
                    return "Unknown"

            tqdm.pandas(desc="Inferring author gender", file=sys.stdout)
            authors_df["author_gender_group"] = authors_df["name"].progress_apply(get_gender)

            # Upload gender mapping back to BigQuery
            table_id = f"{self.project_id}.books.goodreads_author_gender_map"
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            job = self.client.load_table_from_dataframe(authors_df, table_id, job_config=job_config)
            job.result()  # Wait for upload to complete
            self.logger.info(f"Uploaded {len(authors_df)} rows to {table_id}")
            self.logger.info("Uploaded gender map to books.goodreads_author_gender_map")

        except Exception as e:
            self.logger.error(f"Error creating author gender map: {e}", exc_info=True)


def main():
    """
    Main entry point for the data cleaning script.
    
    This function is called by the Airflow DAG to execute the data cleaning pipeline.
    It creates a DataCleaning instance and runs the complete cleaning process.
    """
    data_cleaner = DataCleaning()
    data_cleaner.run()

if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()