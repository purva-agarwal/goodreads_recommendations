import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery, storage
from data_cleaning import clean_goodreads_df_parallel

# ---------------------------------------------------------------------
# 1️⃣ Credentials (absolute path)
# ---------------------------------------------------------------------
def connect_bigquery():
    if os.path.exists("gcp_credentials.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"
# ---------------------------------------------------------------------
# 2️⃣ Logging configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("🚀 Goodreads Data Pipeline started")

# ---------------------------------------------------------------------
# 3️⃣ BigQuery query runner
# ---------------------------------------------------------------------
def run_query(query: str):
    connect_bigquery()
    logging.info("🚀 Starting BigQuery query...")
    client = bigquery.Client(project="recommendation-system-475301")
    job = client.query(query)
    logging.info("⏳ Waiting for query to complete...")
    df = job.result().to_dataframe(create_bqstorage_client=False)
    logging.info(f"📥 Query finished. Retrieved {len(df)} rows.")
    return df

# ---------------------------------------------------------------------
# 4️⃣ Upload to GCS
# ---------------------------------------------------------------------
def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    logging.info(f"✅ Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")

# ---------------------------------------------------------------------
# 5️⃣ Process a single table
# ---------------------------------------------------------------------
def process_table_parallel(table_name, query, out_dir="../data/processed", upload=False, bucket_name=None):
    start_time = time.time()
    logging.info(f"\n🚧 Processing table: {table_name}")

    # Step 1: download
    df = run_query(query)
    logging.info(f"✅ Download complete for {table_name} — cleaning now...")

    # Step 2: clean
    file_name = f"{table_name}_cleaned.parquet"
    cleaned_df = clean_goodreads_df_parallel(df, save=True, out_dir=out_dir, file_name=file_name)

    # Step 3: optional upload
    if upload and bucket_name:
        local_path = os.path.join(out_dir, file_name)
        destination_blob = f"goodreads/{table_name}/{file_name}"
        upload_to_gcs(local_path, bucket_name, destination_blob)

    elapsed = time.time() - start_time
    logging.info(f"⏱️ Finished {table_name} in {elapsed:.2f} seconds.\n")
    return cleaned_df

# ---------------------------------------------------------------------
# 6️⃣ Main entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = "../data/processed"
    os.makedirs(out_dir, exist_ok=True)
    bucket_name = "cleaned_data"

    # Smaller LIMIT for testing (you can increase later)
    tables = {
        # "goodreads_books": """
        #     SELECT *
        #     FROM `recommendation-system-475301.books.goodreads_books_mystery_thriller_crime`
        # """,
        "goodreads_interactions": """
            SELECT *
            FROM `recommendation-system-475301.books.goodreads_interactions_mystery_thriller_crime`
        """
    }

    results = {}
    with ThreadPoolExecutor(max_workers=len(tables)) as executor:
        future_to_table = {
            executor.submit(process_table_parallel, table, query, out_dir, False, bucket_name): table
            for table, query in tables.items()
        }

        for future in as_completed(future_to_table):
            table_name = future_to_table[future]
            try:
                results[table_name] = future.result()
            except Exception as e:
                logging.error(f"❌ Error processing table {table_name}: {e}")

    logging.info("✅ All tables processed successfully!")