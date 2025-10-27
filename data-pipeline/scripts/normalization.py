# goodreads_normalization.py

import os
import logging
from google.cloud import bigquery
from datetime import datetime

class GoodreadsNormalization:

    def __init__(self):
        # Set Google credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME", ".") + "/gcp_credentials.json"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
        self.table = f"{self.project_id}.{self.dataset_id}.goodreads_features_cleaned"
        self.backup_table = f"{self.project_id}.{self.dataset_id}.goodreads_features_cleaned_backup"

    def create_backup(self):
        """Create a backup of the original table."""
        try:
            self.logger.info(f"Creating backup table: {self.backup_table}")
            query = f"""
            CREATE OR REPLACE TABLE `{self.backup_table}` AS
            SELECT * FROM `{self.table}`;
            """
            self.client.query(query).result()
            self.logger.info("Backup created successfully.")
        except Exception as e:
            self.logger.error("Error creating backup table.", exc_info=True)
            raise

    def log_transform_features(self):
        """Apply log transformations to skewed numeric features."""
        try:
            self.logger.info("Applying log transformations to skewed features...")
            query = f"""
            UPDATE `{self.table}`
            SET
              popularity_score = CAST(LN(popularity_score + 1) AS INT64),
              user_activity_count = CAST(LN(user_activity_count + 1) AS INT64),
              description_length = CAST(LN(description_length + 1) AS INT64),
              num_books_read = CASE
                  WHEN num_books_read > 0 THEN CAST(LN(num_books_read + 1) AS INT64)
                  ELSE num_books_read
              END,
              user_days_to_read = CASE
                  WHEN user_days_to_read > 0 THEN CAST(LN(user_days_to_read) AS INT64)
                  ELSE NULL
              END,
              ratings_count = CASE
                  WHEN ratings_count > 0 THEN CAST(LN(ratings_count + 1) AS INT64)
                  ELSE ratings_count
              END,
              num_pages = CASE
                  WHEN num_pages > 0 THEN CAST(LN(num_pages + 1) AS INT64)
                  ELSE num_pages
              END
            WHERE TRUE;
            """
            self.client.query(query).result()
            self.logger.info("Log transformations applied successfully.")
        except Exception as e:
            self.logger.error("Error applying log transformations.", exc_info=True)
            raise

    def normalize_user_ratings(self):
        """Apply user-centered rating normalization."""
        try:
            self.logger.info("Applying user-centered rating normalization...")
            alter_query = f"""
            ALTER TABLE `{self.table}`
            ALTER COLUMN rating SET DATA TYPE FLOAT64;
            """
            self.client.query(alter_query).result()

            update_query = f"""
            UPDATE `{self.table}`
            SET rating = rating - avg_rating_given
            WHERE TRUE;
            """
            self.client.query(update_query).result()
            self.logger.info("User-centered rating normalization applied successfully.")
        except Exception as e:
            self.logger.error("Error normalizing ratings.", exc_info=True)
            raise

    def run(self):
        """Run the full normalization pipeline."""
        self.logger.info("="*60)
        self.logger.info("GOODREADS FEATURE NORMALIZATION PIPELINE")
        self.logger.info(f"Started at: {datetime.now()}")
        self.logger.info("="*60)

        self.create_backup()

        self.log_transform_features()

        self.normalize_user_ratings()

        self.logger.info(f"Completed at: {datetime.now()}")
        self.logger.info("="*60)


def main():
    normalizer = GoodreadsNormalization()
    normalizer.run()


if __name__ == "__main__":
    main()
