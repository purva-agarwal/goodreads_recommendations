



import os
from google.cloud import bigquery
from datapipeline.scripts.logger_setup import get_logger

class GoodreadsNormalizedSplitter:

    def __init__(self):

        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME") + "/gcp_credentials.json"

        self.logger = get_logger("normalized_split")
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

 
        self.source_table = f"{self.project_id}.{self.dataset_id}.goodreads_features"
        self.train_table = f"{self.project_id}.{self.dataset_id}.goodreads_train_set"
        self.val_table = f"{self.project_id}.{self.dataset_id}.goodreads_validation_set"
        self.test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"

    def run_split_in_bq(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train/validation/test in BigQuery using user_id_clean column."""

        self.logger.info("Starting BigQuery-based data split...")

        # Train
        train_query = f"""
        CREATE OR REPLACE TABLE `{self.train_table}` AS
        SELECT *
        FROM `{self.source_table}`
        WHERE MOD(ABS(FARM_FINGERPRINT(CAST(user_id_clean AS STRING))), 100) < {int(train_ratio*100)};
        """
        self.client.query(train_query).result()
        self.logger.info(f"Train table created: {self.train_table}")

        # Validation
        val_query = f"""
        CREATE OR REPLACE TABLE `{self.val_table}` AS
        SELECT *
        FROM `{self.source_table}`
        WHERE MOD(ABS(FARM_FINGERPRINT(CAST(user_id_clean AS STRING))), 100)
              BETWEEN {int(train_ratio*100)} AND {int((train_ratio+val_ratio)*100)-1};
        """
        self.client.query(val_query).result()
        self.logger.info(f"Validation table created: {self.val_table}")

        # Test
        test_query = f"""
        CREATE OR REPLACE TABLE `{self.test_table}` AS
        SELECT *
        FROM `{self.source_table}`
        WHERE MOD(ABS(FARM_FINGERPRINT(CAST(user_id_clean AS STRING))), 100) >= {int((train_ratio+val_ratio)*100)};
        """
        self.client.query(test_query).result()
        self.logger.info(f"Test table created: {self.test_table}")

        self.logger.info("BigQuery-based split completed successfully.")

    def run(self):
        self.run_split_in_bq()


def main():
    splitter = GoodreadsNormalizedSplitter()
    splitter.run()


if __name__ == "__main__":
    main()
