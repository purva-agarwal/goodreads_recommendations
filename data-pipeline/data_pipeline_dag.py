from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime, timedelta
import os
from airflow.utils.email import send_email


from scripts.bigquery_full_data_cleaning import main as data_cleaning_main
from scripts.feature_engineering import main as feature_engg_main
from scripts.normalization import main as normalization_main
from scripts.anomaly_detection import main as anomaly_detection_main

default_args = {
    'owner': 'Arpita/Shivani',
    'start_date': datetime(2025, 1, 18),
    'retries': 0,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False,
    'email_on_retry': False,
    
}
def send_failure_email(context):
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id
    task_id = task_instance.task_id
    execution_date = context.get('execution_date')
    log_url = task_instance.log_url

    subject = f"[Airflow] DAG {dag_id} Failed: Task {task_id}"
    html_content = f"""
    <p>DAG <b>{dag_id}</b> failed for task <b>{task_id}</b> on {execution_date}.</p>
    <p>Check logs: <a href="{log_url}">Click here</a></p>
    """
    send_email(to="7d936ad4-351b-4493-99a7-110ed7b6b2f6@emailhook.site", subject=subject, html_content=html_content)

def send_success_email(context):
    dag_id = context.get('dag').dag_id
    execution_date = context.get('execution_date')
    subject = f"[Airflow] DAG {dag_id} Succeeded"
    html_content = f"""
    <p>DAG <b>{dag_id}</b> succeeded for execution date {execution_date}.</p>
    """
    send_email(to="7d936ad4-351b-4493-99a7-110ed7b6b2f6@emailhook.site", subject=subject, html_content=html_content)
    
def log_query_results(**kwargs):
    ti = kwargs['ti']
    job_id = ti.xcom_pull(task_ids='read_data_from_bigquery')

    from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
    import logging

    hook = BigQueryHook(gcp_conn_id="goodreads_conn")
    client = hook.get_client()

    query_job = client.get_job(job_id)
    rows = list(query_job.result())

    logging.info("Query Results:")
    for row in rows:
        logging.info(dict(row))


with DAG(
    dag_id='goodreads_recommendation_pipeline',
    default_args=default_args,
    description='Goodreads Recommendation System Data Pipeline',
    catchup=False,
    on_failure_callback=send_failure_email, 
    on_success_callback=send_success_email,  
) as dag:
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"

    start = EmptyOperator(task_id='start')

    data_reading_task = BigQueryInsertJobOperator(
        task_id='read_data_from_bigquery',
        configuration={
            "query": {
                "query": """
                    SELECT 
                        'books' as table_type,
                        COUNT(*) as record_count
                    FROM `recommendation-system-475301.books.goodreads_books_mystery_thriller_crime`
                    UNION ALL
                    SELECT 
                        'interactions' as table_type,
                        COUNT(*) as record_count
                    FROM `recommendation-system-475301.books.goodreads_interactions_mystery_thriller_crime`
                """,
                "useLegacySql": False,
            }
        },
        gcp_conn_id='goodreads_conn',
    )

    log_results_task = PythonOperator(
        task_id='log_bq_results',
        python_callable=log_query_results,
    )

    data_validation_task = PythonOperator(
        task_id='validate_data_quality',
        python_callable=anomaly_detection_main,
        doc_md="""
        ## Data Validation Task
        Simple data quality checks:
        - Required columns exist
        - Data ranges are valid
        - Missing values within limits
        - Stops pipeline if critical issues found
        """
    )

    data_cleaning_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=data_cleaning_main,
    )
    
    post_cleaning_validation_task = PythonOperator(
        task_id='validate_cleaned_data',
        python_callable=anomaly_detection_main,
        doc_md="""
        ## Post-Cleaning Validation Task
        Validates data quality after cleaning:
        - Ensures cleaning process worked correctly
        - Checks for any new data quality issues
        - Validates cleaned data meets requirements
        """
    )
    
    feature_engg_task = PythonOperator(
        task_id='feature_engg_data',
        python_callable=feature_engg_main,
    )
    normalization_task = PythonOperator(
        task_id='normalize_data',
        python_callable=normalization_main,
    )

    end = EmptyOperator(task_id='end')

    start >> data_reading_task >> log_results_task >> data_validation_task >> data_cleaning_task >> post_cleaning_validation_task >> feature_engg_task >> normalization_task >> end
