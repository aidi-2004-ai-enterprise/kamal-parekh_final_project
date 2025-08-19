

import os
from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.python import PythonOperator
# from airflow.providers.smtp.operators.email import EmailOperator # Uncomment for email notification

# Required for model training and GCS interaction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from google.cloud import bigquery
from google.cloud import storage

# --- Configuration ---
# Replace with your GCP project ID
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "zeta-rush-468118-e7")
# BigQuery dataset to store intermediate data (must exist or be created manually)
DATASET_ID = "ml_pipeline"
# Table name for prepared data in BigQuery
TABLE_ID = "london_bicycles_prepared_data"
# GCS bucket to store model and metrics (must exist)
GCS_BUCKET = "us-central1-ml-pipeline-env-74be3d9e-bucket" # e.g., "my-composer-ml-models"
# Email recipient for notifications (uncomment EmailOperator to use)
EMAIL_RECIPIENT = "kamalparekh515@gmail.com"

# --- Default Airflow Arguments ---
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- DAG Definition ---
with DAG(
    dag_id="cloud_composer_ml_pipeline",
    default_args=default_args,
    description="Automated ML pipeline for London Bicycles data.",
    schedule_interval=timedelta(weeks=1),  # Weekly schedule
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ml", "bigquery", "gcs", "composer"],
) as dag:
    # --- Task 1: Data Extraction from BigQuery ---
    # This task extracts selected features and target from the public dataset
    # and saves them into a new table in your project's BigQuery dataset.
    extract_data_to_bq = BigQueryInsertJobOperator(
        task_id="extract_data_to_bigquery",
        configuration={
            "query": {
                "query": f"""
                    SELECT
                        duration,
                        start_station_id,
                        end_station_id,
                        EXTRACT(HOUR FROM start_date) AS hour_of_day,
                        EXTRACT(DAYOFWEEK FROM start_date) AS day_of_week
                    FROM
                        `bigquery-public-data.london_bicycles.cycle_hire`
                    WHERE
                        duration IS NOT NULL AND duration > 60 AND duration < 10000
                        AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL
                    LIMIT 50000 -- Limit for simplicity and faster execution during demo
                """,
                "useLegacySql": False,
                "destinationTable": {
                    "projectId": PROJECT_ID,
                    "datasetId": DATASET_ID,
                    "tableId": TABLE_ID,
                },
                "writeDisposition": "WRITE_TRUNCATE",  # Overwrite table if it exists
            }
        },
        gcp_conn_id="google_cloud_default", # Ensure this connection is configured
    )

    # --- Task 2: Model Training & Persistence ---
    # This Python function connects to BigQuery, loads the prepared data,
    # trains a simple Linear Regression model, evaluates it, and then saves
    # the trained model and evaluation metrics to a GCS bucket.
    def train_model_and_persist_func(project_id, dataset_id, table_id, gcs_bucket, **kwargs):
        logging.info(f"Starting model training for project: {project_id}, dataset: {dataset_id}, table: {table_id}")

        # Initialize BigQuery client
        bq_client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        # Fetch data from BigQuery
        query = f"SELECT * FROM `{table_ref}`"
        df = bq_client.query(query).to_dataframe()
        logging.info(f"Fetched {len(df)} rows from BigQuery.")

        # --- Data Preprocessing ---
        # Handle categorical features by converting them to 'category' dtype
        categorical_cols = ['start_station_id', 'end_station_id', 'hour_of_day', 'day_of_week']
        for col in categorical_cols:
            df[col] = df[col].astype('category')

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Define features (X) and target (y)
        features = df_encoded.drop("duration", axis=1)
        target = df_encoded["duration"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # --- Model Training ---
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        # --- Model Evaluation ---
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Model Evaluation - Mean Absolute Error (MAE): {mae:.2f}")
        logging.info(f"Model Evaluation - Mean Squared Error (MSE): {mse:.2f}")

        # --- Model and Metrics Persistence to GCS ---
        gcs_client = storage.Client(project=project_id)
        bucket = gcs_client.bucket(gcs_bucket)

        # Define GCS object names with a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"models/london_bicycles_linear_regression_model_{timestamp}.joblib"
        metrics_filename = f"metrics/london_bicycles_metrics_{timestamp}.json"

        # Save model to GCS
        model_blob = bucket.blob(model_filename)
        # Create a temporary file to save the model, then upload it
        temp_model_path = "/tmp/model.joblib"
        joblib.dump(model, temp_model_path)
        model_blob.upload_from_filename(temp_model_path)
        logging.info(f"Model saved to gs://{gcs_bucket}/{model_filename}")

        # Save metrics to GCS
        metrics = {"mae": mae, "mse": mse, "timestamp": timestamp}
        metrics_blob = bucket.blob(metrics_filename)
        # Create a temporary file to save metrics, then upload it
        temp_metrics_path = "/tmp/metrics.json"
        import json
        with open(temp_metrics_path, "w") as f:
            json.dump(metrics, f)
        metrics_blob.upload_from_filename(temp_metrics_path)
        logging.info(f"Metrics saved to gs://{gcs_bucket}/{metrics_filename}")

        # Push metrics summary to XCom for the next task
        metrics_summary = f"MAE: {mae:.2f}, MSE: {mse:.2f}. Model: gs://{gcs_bucket}/{model_filename}"
        kwargs["ti"].xcom_push(key="model_metrics_summary", value=metrics_summary)

    train_model_and_persist = PythonOperator(
        task_id="train_model_and_persist",
        python_callable=train_model_and_persist_func,
        op_kwargs={
            "project_id": PROJECT_ID,
            "dataset_id": DATASET_ID,
            "table_id": TABLE_ID,
            "gcs_bucket": GCS_BUCKET,
        },
    )

    # --- Task 3: Notification ---
    # This task logs a completion message to Cloud Logging and can optionally
    # send an email notification with the model's performance summary.
    def log_completion_func(**kwargs):
        metrics_summary = kwargs["ti"].xcom_pull(key="model_metrics_summary", task_ids="train_model_and_persist")
        if not metrics_summary:
            metrics_summary = "Metrics not available."
        logging.info(f"🎉 Model training complete! Check GCS for saved model and metrics. Summary: {metrics_summary}")

    log_completion = PythonOperator(
        task_id="log_completion_message",
        python_callable=log_completion_func,
    )
