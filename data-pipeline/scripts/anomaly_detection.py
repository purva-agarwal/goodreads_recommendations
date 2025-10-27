"""
Anomaly Detection using Great Expectations for Goodreads Data Pipeline
Based on DataCamp tutorial approach
"""

import os
import logging
import pandas as pd
import great_expectations as gx
from google.cloud import bigquery
from airflow.utils.email import send_email

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data_quality(use_cleaned_tables=False):
    """
    Data quality validation using Great Expectations
    Args:
        use_cleaned_tables (bool): If True, validate cleaned tables; if False, validate source tables
    """
    try:
        # Setup BigQuery client
        project_id = "recommendation-system-475301"
        dataset = "books"
        
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            client = bigquery.Client(project=project_id)
        else:
            logger.error("GCP credentials not found")
            raise Exception("GCP credentials not configured")
        
        validation_type = "cleaned" if use_cleaned_tables else "source"
        logger.info(f"Starting Great Expectations data validation for {validation_type} tables...")
        
        # Initialize Great Expectations context
        context = gx.get_context()
        
        # Validate books table
        books_success = validate_books_with_gx(client, project_id, dataset, context, use_cleaned_tables)
        
        # Validate interactions table  
        interactions_success = validate_interactions_with_gx(client, project_id, dataset, context, use_cleaned_tables)
        
        # Check overall success
        if not books_success or not interactions_success:
            send_failure_email(f"Data validation failed for {validation_type} tables - check logs for details")
            raise Exception(f"Data validation failed for {validation_type} tables - critical issues found")
        
        logger.info(f"All {validation_type} data quality validations passed")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise

def get_table_structure(client, project_id, dataset, table_name):
    """
    Get table structure from BigQuery INFORMATION_SCHEMA
    """
    try:
        columns_info = client.query(f"""
            SELECT column_name, data_type
            FROM `{project_id}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """).to_dataframe(create_bqstorage_client=False)
        
        logger.info(f"Retrieved {len(columns_info)} columns for table {table_name}")
        return columns_info
    except Exception as e:
        logger.error(f"Error fetching table structure for {table_name}: {e}")
        return None

def validate_books_with_gx(client, project_id, dataset, context, use_cleaned_tables=False):
    """
    Validate books table using Great Expectations with dynamic schema
    Args:
        use_cleaned_tables (bool): If True, validate cleaned table; if False, validate source table
    """
    try:
        # Choose table based on validation type
        table_name = "goodreads_books_cleaned" if use_cleaned_tables else "goodreads_books_mystery_thriller_crime"
        
        # Get table structure
        columns_info = get_table_structure(client, project_id, dataset, table_name)
        if columns_info is None:
            return False
        
        # Get sample data
        query = f"""
        SELECT * FROM `{project_id}.{dataset}.{table_name}`
        """
        df = client.query(query).to_dataframe()
        
        if df.empty:
            logger.error("Books table is empty")
            return False
        
        logger.info(f"Validating books table with Great Expectations: {len(df)} rows")
        
        # Create datasource configuration
        datasource_config = {
            "name": "books_datasource",
            "class_name": "Datasource",
            "execution_engine": {
                "class_name": "PandasExecutionEngine"
            },
            "data_connectors": {
                "default_runtime_data_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier_name"]
                }
            }
        }
        
        # Add datasource
        try:
            context.add_datasource(**datasource_config)
        except:
            pass  # Datasource might already exist
        
        # Create expectation suite
        suite = context.create_expectation_suite("books_expectations", overwrite_existing=True)
        
        # Create batch request
        batch_request = {
            "datasource_name": "books_datasource",
            "data_connector_name": "default_runtime_data_connector",
            "data_asset_name": "books_data",
            "runtime_parameters": {
                "batch_data": df
            },
            "batch_identifiers": {
                "default_identifier_name": "default_identifier"
            }
        }
        
        # Get validator
        validator = context.get_validator(
            batch_request=batch_request, 
            expectation_suite_name="books_expectations"
        )
        
        # Add expectations based on actual table structure
        for _, row in columns_info.iterrows():
            col_name = row['column_name']
            data_type = row['data_type']
            
            # Column existence
            validator.expect_column_to_exist(col_name)
            
            # Data type expectations based on BigQuery types
            if data_type in ('STRING', 'CHAR', 'TEXT'):
                validator.expect_column_values_to_be_of_type(col_name, "str")
            elif data_type == 'INT64':
                validator.expect_column_values_to_be_of_type(col_name, "int64")
            elif data_type == 'FLOAT64':
                validator.expect_column_values_to_be_of_type(col_name, "float64")
            elif data_type == 'BOOL':
                validator.expect_column_values_to_be_of_type(col_name, "bool")
        
        # Business rule expectations for specific columns
        if 'average_rating' in df.columns:
            validator.expect_column_values_to_be_between("average_rating", min_value=0, max_value=5)
        if 'publication_year' in df.columns:
            validator.expect_column_values_to_be_between("publication_year", min_value=1000, max_value=2030)
        if 'num_pages' in df.columns:
            validator.expect_column_values_to_be_between("num_pages", min_value=1, max_value=10000)
        if 'ratings_count' in df.columns:
            validator.expect_column_values_to_be_between("ratings_count", min_value=0)
        if 'publication_month' in df.columns:
            validator.expect_column_values_to_be_between("publication_month", min_value=1, max_value=12)
        if 'publication_day' in df.columns:
            validator.expect_column_values_to_be_between("publication_day", min_value=1, max_value=31)
        
        # Missing value expectations (allow some missing values)
        for _, row in columns_info.iterrows():
            col_name = row['column_name']
            if col_name in df.columns:
                validator.expect_column_values_to_not_be_null(col_name, mostly=0.8)  # Allow 20% missing
        
        # Save expectations
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        # Create checkpoint
        checkpoint = context.add_or_update_checkpoint(
            name="books_checkpoint",
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": "books_expectations"
                }
            ]
        )
        
        # Run validation
        checkpoint_result = checkpoint.run()
        
        if checkpoint_result["success"]:
            logger.info("Books table validation passed")
            return True
        else:
            logger.error("Books table validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Error validating books table: {e}")
        return False

def validate_interactions_with_gx(client, project_id, dataset, context, use_cleaned_tables=False):
    """
    Validate interactions table using Great Expectations with dynamic schema
    Args:
        use_cleaned_tables (bool): If True, validate cleaned table; if False, validate source table
    """
    try:
        # Choose table based on validation type
        table_name = "goodreads_interactions_cleaned" if use_cleaned_tables else "goodreads_interactions_mystery_thriller_crime"
        
        # Get table structure
        columns_info = get_table_structure(client, project_id, dataset, table_name)
        if columns_info is None:
            return False
        
        # Get sample data
        query = f"""
        SELECT * FROM `{project_id}.{dataset}.{table_name}`
        """
        df = client.query(query).to_dataframe()
        
        if df.empty:
            logger.error("Interactions table is empty")
            return False
        
        logger.info(f"Validating interactions table with Great Expectations: {len(df)} rows")
        
        # Create datasource configuration
        datasource_config = {
            "name": "interactions_datasource",
            "class_name": "Datasource",
            "execution_engine": {
                "class_name": "PandasExecutionEngine"
            },
            "data_connectors": {
                "default_runtime_data_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier_name"]
                }
            }
        }
        
        # Add datasource
        try:
            context.add_datasource(**datasource_config)
        except:
            pass  # Datasource might already exist
        
        # Create expectation suite
        suite = context.create_expectation_suite("interactions_expectations", overwrite_existing=True)
        
        # Create batch request
        batch_request = {
            "datasource_name": "interactions_datasource",
            "data_connector_name": "default_runtime_data_connector",
            "data_asset_name": "interactions_data",
            "runtime_parameters": {
                "batch_data": df
            },
            "batch_identifiers": {
                "default_identifier_name": "default_identifier"
            }
        }
        
        # Get validator
        validator = context.get_validator(
            batch_request=batch_request, 
            expectation_suite_name="interactions_expectations"
        )
        
        # Add expectations based on actual table structure
        for _, row in columns_info.iterrows():
            col_name = row['column_name']
            data_type = row['data_type']
            
            # Column existence
            validator.expect_column_to_exist(col_name)
            
            # Data type expectations based on BigQuery types
            if data_type in ('STRING', 'CHAR', 'TEXT'):
                validator.expect_column_values_to_be_of_type(col_name, "str")
            elif data_type == 'INT64':
                validator.expect_column_values_to_be_of_type(col_name, "int64")
            elif data_type == 'FLOAT64':
                validator.expect_column_values_to_be_of_type(col_name, "float64")
            elif data_type == 'BOOL':
                validator.expect_column_values_to_be_of_type(col_name, "bool")
        
        # Business rule expectations for specific columns
        if 'rating' in df.columns:
            validator.expect_column_values_to_be_between("rating", min_value=0, max_value=5)
        if 'book_id' in df.columns:
            validator.expect_column_values_to_be_between("book_id", min_value=1)
        
        # Missing value expectations (allow some missing values)
        for _, row in columns_info.iterrows():
            col_name = row['column_name']
            if col_name in df.columns:
                validator.expect_column_values_to_not_be_null(col_name, mostly=0.8)  # Allow 20% missing
        
        # Save expectations
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        # Create checkpoint
        checkpoint = context.add_or_update_checkpoint(
            name="interactions_checkpoint",
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": "interactions_expectations"
                }
            ]
        )
        
        # Run validation
        checkpoint_result = checkpoint.run()
        
        if checkpoint_result["success"]:
            logger.info("Interactions table validation passed")
            return True
        else:
            logger.error("Interactions table validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Error validating interactions table: {e}")
        return False

def send_failure_email(message):
    """
    Send email notification for validation failures
    """
    try:
        subject = "[CRITICAL] Data Validation Failed - Goodreads Pipeline"
        
        html_content = f"""
        <h2>Data Validation Failure</h2>
        <p><strong>Pipeline:</strong> Goodreads Recommendation System</p>
        <p><strong>Status:</strong> FAILED - Pipeline stopped</p>
        <p><strong>Error:</strong> {message}</p>
        
        <p><strong>Action Required:</strong> Please investigate and fix the data quality issues before re-running the pipeline.</p>
        <p><em>This is an automated alert from the Goodreads Data Pipeline.</em></p>
        """
        
        send_email(
            to="7d936ad4-351b-4493-99a7-110ed7b6b2f6@emailhook.site",
            subject=subject,
            html_content=html_content
        )
        
        logger.info("Validation failure email sent")
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def main():
    """
    Main function called by Airflow DAG
    For pre-cleaning validation, use source tables
    For post-cleaning validation, use cleaned tables
    """
    # Check if this is post-cleaning validation by looking for cleaned tables
    try:
        project_id = "recommendation-system-475301"
        dataset = "books"
        
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            client = bigquery.Client(project=project_id)
        else:
            logger.error("GCP credentials not found")
            raise Exception("GCP credentials not configured")
        
        # Check if cleaned tables exist to determine validation type
        cleaned_books_exists = check_table_exists(client, project_id, dataset, "goodreads_books_cleaned")
        cleaned_interactions_exists = check_table_exists(client, project_id, dataset, "goodreads_interactions_cleaned")
        
        use_cleaned_tables = cleaned_books_exists and cleaned_interactions_exists
        
        validation_type = "post-cleaning" if use_cleaned_tables else "pre-cleaning"
        logger.info(f"Running {validation_type} validation...")
        
        return validate_data_quality(use_cleaned_tables)
        
    except Exception as e:
        logger.error(f"Error determining validation type: {e}")
        # Default to pre-cleaning validation
        return validate_data_quality(use_cleaned_tables=False)

def check_table_exists(client, project_id, dataset, table_name):
    """
    Check if a table exists in BigQuery
    """
    try:
        query = f"""
        SELECT COUNT(*) as count
        FROM `{project_id}.{dataset}.INFORMATION_SCHEMA.TABLES`
        WHERE table_name = '{table_name}'
        """
        result = client.query(query).to_dataframe()
        return result['count'].iloc[0] > 0
    except Exception as e:
        logger.error(f"Error checking if table {table_name} exists: {e}")
        return False

if __name__ == "__main__":
    main()