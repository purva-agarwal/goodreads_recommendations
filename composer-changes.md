# Goodreads Book Recommendation System - MLOps Pipeline


### Key Features
- **Automated Data Pipeline**: End-to-end data processing from raw data to ML-ready features
- **Cloud-Native Architecture**: Built on Google Cloud Platform (BigQuery, Cloud Composer, GCS)
- **Data Quality Validation**: Comprehensive anomaly detection and data validation
- **Feature Engineering**: Advanced feature extraction for collaborative filtering
- **Scalable Infrastructure**: Handles millions of book-user interactions
- **Monitoring & Alerting**: Email notifications for pipeline failures/success

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   Google Cloud Platform                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌─────────────────┐            │
│  │   BigQuery   │◄─────│ Cloud Composer  │            │
│  │              │      │   (Airflow)     │            │
│  └──────────────┘      └─────────────────┘            │
│         ▲                       │                       │
│         │                       ▼                       │
│  ┌──────────────┐      ┌─────────────────┐            │
│  │     GCS      │◄─────│   Data Pipeline │            │
│  │   Storage    │      │    Scripts      │            │
│  └──────────────┘      └─────────────────┘            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```


## 🚀 Getting Started

### Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and authenticated
- Python 3.8+
- Apache Airflow (for local development)

### GCP Setup

1. **Set up Google Cloud Project**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable composer.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable storage-component.googleapis.com
```

2. **Create Cloud Composer Environment**
```bash
gcloud composer environments create recommendation-system-airflow \
    --location us-central1 \
    --python-version 3
```

3. **Get Environment Information**
```bash
# Get bucket name
BUCKET=$(gcloud composer environments describe recommendation-system-airflow \
    --location us-central1 \
    --format="value(config.dagGcsPrefix)" | sed 's|/dags||')

echo $BUCKET

# Get Airflow UI URL
gcloud composer environments describe recommendation-system-airflow \
    --location us-central1 \
    --format="value(config.airflowUri)"
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/goodreads_recommendations.git
cd goodreads_recommendations
```

2. **Upload DAG to Cloud Composer**
```bash
gsutil cp dags/data_pipeline_dag.py ${BUCKET}dags/
```

3. **Upload pipeline scripts**
```bash
gsutil -m cp -r data/datapipeline ${BUCKET}data/
```

4. **Install Python dependencies**
```bash
gcloud composer environments update recommendation-system-airflow \
    --location us-central1 \
    --update-pypi-packages-from-file requirements.txt
```

5. **Configure email notifications**
```bash
# Create SMTP connection
gcloud composer environments run recommendation-system-airflow \
    --location us-central1 \
    connections add -- \
    smtp_default \
    --conn-type email \
    --conn-host smtp.gmail.com \
    --conn-login your.email@gmail.com \
    --conn-password your_app_password \
    --conn-port 587

# Set notification email
gcloud composer environments run recommendation-system-airflow \
    --location us-central1 \
    variables set -- notification_email your.email@gmail.com
```


## 🏃 Running the Pipeline

### Via Airflow UI
1. Access the Airflow web interface
2. Navigate to DAGs
3. Find `goodreads_recommendation_pipeline`
4. Click "Trigger DAG"

### Via CLI
```bash
gcloud composer environments run recommendation-system-airflow \
    --location us-central1 \
    dags trigger -- goodreads_recommendation_pipeline
```

### Monitor Execution
```bash
# Check DAG status
gcloud composer environments run recommendation-system-airflow \
    --location us-central1 \
    dags state -- goodreads_recommendation_pipeline EXECUTION_DATE

# View logs
gsutil cat ${BUCKET}logs/goodreads_recommendation_pipeline/*/
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Ensure `__init__.py` files exist in all directories
   - Verify path configuration in DAG file
   - Check files are uploaded to correct GCS location

2. **Authentication Errors**
   - Don't set `GOOGLE_APPLICATION_CREDENTIALS` in Cloud Composer
   - Cloud Composer handles authentication automatically

3. **DAG Not Appearing**
   - Wait 3-5 minutes after upload
   - Check for syntax errors: `python dags/data_pipeline_dag.py`
   - View import errors:
```bash
     gcloud composer environments run recommendation-system-airflow \
         --location us-central1 \
         dags list-import-errors
```

4. **Email Not Working**
   - Verify SMTP connection is configured
   - Check notification_email variable is set
   - Ensure app password (not regular password) is used for Gmail

### Debug Commands
```bash
# List all DAGs
gcloud composer environments run recommendation-system-airflow \
    --location us-central1 \
    dags list

# Check connections
gcloud composer environments run recommendation-system-airflow \
    --location us-central1 \
    connections list

# View environment logs
gcloud logging read "resource.type=cloud_composer_environment" \
    --limit 50 \
    --format json
```

## 📈 Performance Optimization

- **BigQuery**: Use partitioned tables for large datasets
- **Airflow**: Adjust parallelism and worker count
- **GCS**: Use `-m` flag for parallel uploads/downloads
- **Memory**: Monitor Cloud Composer node memory usage

## 🧪 Testing

Run tests locally:
```bash
# All tests
pytest datapipeline/tests/

# Specific test
pytest datapipeline/tests/test_data_cleaning.py

# With coverage
pytest --cov=datapipeline datapipeline/tests/
```

## 📝 Dependencies

Key Python packages:
- `apache-airflow`
- `google-cloud-bigquery`
- `google-cloud-storage`
- `pandas`
- `numpy`
- `scikit-learn`
- `gender-guesser`
- `pytest`

## Important commands
```bash
# Copy to tests
gsutil cp test_feature_engineering.py gs://us-central1-recommendation--e28d8ad5-bucket/data/datapipeline/tests/

# Copy to dags
gsutil cp data_pipeline_dag.py gs://us-central1-recommendation--e28d8ad5-bucket/dags

# Copy to scripts
gsutil cp promote_staging_tables.py gs://us-central1-recommendation--e28d8ad5-bucket/data/datapipeline/scripts/

# Setup SMTP
gcloud composer environments run recommendation-system-airflow --location us-central1 connections add -- smtp_default --conn-type email --conn-host smtp.gmail.com --conn-login husky.mlops@gmail.com --conn-password pwd --conn-port 587 --conn-extra "{\"smtp_starttls\": true, \"smtp_ssl\": false, \"smtp_mail_from\": \"husky.mlops@gmail.com\"}"
```