"""
Staging Table Promotion Module for Goodreads Data Pipeline

This module handles the promotion of staging tables to production by replacing
the production tables with the cleaned and processed staging data. It ensures
a clean transition from staging to production environment.

Key Features:
- Promotes staging tables to production tables
- Replaces existing production tables with staging data
- Cleans up staging tables after successful promotion
- Provides logging for promotion operations

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datapipeline.scripts.logger_setup import get_logger

class StagingTablePromoter:

    def __init__(self):
        """
        Initialize the StagingTablePromoter class with BigQuery client and configuration.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration for promotion operations
        - BigQuery client and project information
        - Dataset reference for table operations
        """
        # Set Google Application Credentials for BigQuery access
        # Uses AIRFLOW_HOME environment variable to locate credentials file
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME", ".") + "/gcp_credentials.json"

        # Initialize logging for promotion operations
        self.logger = get_logger("promote_staging_tables")

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
    
    def promote_staging_tables(self):
        """
        Promote staging tables to production by replacing the production tables.
        
        This method performs the following operations for each staging table:
        1. Creates or replaces the production table with staging data
        2. Drops the staging table after successful promotion
        
        The promotion ensures a clean transition from staging to production
        environment with proper cleanup of temporary staging tables.
        """
        try:
            # Define mapping of staging tables to production tables
            staging_tables = {
                "goodreads_books_cleaned_staging": "goodreads_books_cleaned",
                "goodreads_interactions_cleaned_staging": "goodreads_interactions_cleaned",
                "goodreads_features_cleaned_staging": "goodreads_features"
            }

            # Promote each staging table to production
            for staging_table, prod_table in staging_tables.items():
                self.logger.info(f"Promoting {staging_table} to {prod_table}...")
                
                # Create or replace production table with staging data
                query = f"""
                CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.{prod_table}` AS
                SELECT * FROM `{self.project_id}.{self.dataset_id}.{staging_table}`;
                """
                self.client.query(query).result()
                self.logger.info(f"Successfully promoted {staging_table} to {prod_table}.")
            
                # Remove staging table after successful promotion
                self.logger.info(f"Dropping staging table: {staging_table}...")
                drop_query = f"DROP TABLE `{self.project_id}.{self.dataset_id}.{staging_table}`;"
                self.client.query(drop_query).result()
                self.logger.info(f"Successfully dropped staging table: {staging_table}.")

        except Exception as e:
            self.logger.error("Error promoting staging tables.", exc_info=True)
            raise

    def run(self):
        """
        Execute the staging table promotion process.
        
        This method runs the complete promotion pipeline to move staging tables
        to production and clean up temporary staging resources.
        """
        self.promote_staging_tables()

def main():
    """
    Main entry point for the staging table promotion script.
    
    This function is called by the Airflow DAG to execute the table promotion process.
    It creates a StagingTablePromoter instance and runs the promotion pipeline.
    """
    promoter = StagingTablePromoter()
    promoter.run()

if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()