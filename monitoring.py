import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

import mlflow
from mlflow import MlflowClient


@task()
def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=root", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='housing_price'")
        if len(res.fetchall()) == 0:
            conn.execute("create database housing_price;")
            print("Database created")
                        
            with psycopg.connect("host=localhost port=5432 dbname=housing_price user=postgres password=root") as conn:
                conn.execute(create_table_statement)
                print("table created")

@task()
def calculate_metrics_postgresql(curr, raw_data, loaded_model, reference_data):

    num_feat = raw_data.select_dtypes(exclude='object').columns.tolist()

    column_mapping = ColumnMapping(
        target=None,
        prediction='price',
        numerical_features=num_feat,
    )

    report = Report(metrics=[
        ColumnDriftMetric(column_name='price'),
        DatasetDriftMetric(),
    ]
    )
    current_data = raw_data
    current_data['price'] = loaded_model.predict(current_data[num_feat].drop(['ID'], axis=1))

    # Assuming 'report' is an instance of a class that performs some kind of reporting
    # and contains methods like 'run' and 'as_dict'
    report.run(reference_data=reference_data, current_data=raw_data, column_mapping=column_mapping)
    result = report.as_dict()
    
    # Extracting metrics from the result dictionary
    model_drift_score = result['metrics'][0]['result']['drift_score']
    model_drift_detected = result['metrics'][0]['result']['drift_detected']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    dataset_drift_detected = result['metrics'][1]['result']['dataset_drift']

    # Inserting the metrics into the PostgreSQL database
    curr.execute(
        "INSERT INTO housing_metrics (timestamp, model_drift_score, model_drift_detected, num_drifted_columns, dataset_drift_detected) VALUES (%s, %s, %s, %s, %s)",
        (datetime.datetime.now(), model_drift_score, model_drift_detected, num_drifted_columns, dataset_drift_detected)
    )

@flow()
def batch_monitoring_backfill(raw_data, loaded_model, reference_data):
    prep_db()
    with psycopg.connect("host=localhost port=5432 dbname=housing_price user=postgres password=root", autocommit=True) as conn:
        with conn.cursor() as curr:
            calculate_metrics_postgresql(curr, raw_data, loaded_model, reference_data)
        logging.info("data sent")
        print("data sent")

@flow()
def main_task(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, model_registry_name, client):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

    
    prod_model = client.get_latest_versions(model_registry_name, stages=["Production"])[0]
    prod_model_run_id = prod_model.run_id
    logged_model = f'runs:/{prod_model_run_id}/model'
    # loaded_model = mlflow.pyfunc.load_model(logged_model)

    with open('models/lgb.bin', 'rb') as f_in:
        loaded_model = joblib.load(f_in)


    SEND_TIMEOUT = 10
    rand = random.Random()

    create_table_statement = """
    drop table if exists housing_metrics;
    create table housing_metrics(
        timestamp timestamp,
        model_drift_score float,
        model_drift_detected boolean,
        num_drifted_columns integer,
        dataset_drift_detected boolean
    )
    """

    reference_data = pd.read_csv('data/train_data.csv')

    raw_data = pd.read_csv('data/test_data.csv')

    batch_monitoring_backfill(raw_data, loaded_model, reference_data)


if __name__ == '__main__':
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = 'housing-price'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_registry_name = "housing_price"

    client = MlflowClient()

    main_task(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, model_registry_name, client)