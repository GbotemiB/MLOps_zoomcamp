from prefect import task, flow
import mlflow
from mlflow.tracking import MlflowClient

from monitoring import main_task as monitoring
from model_train import run as model_train
from model_registry_update import runner as model_registry

@flow(name="main_flow_run")
def main_run():
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = 'housing-price'
    model_registry_name = 'housing_price'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    data_path = 'data/Housing_dataset_train.csv'
    n_trials = 2

    model_train(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, model_registry_name, client, data_path, n_trials)
    model_registry(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, model_registry_name, client, experiment)
    monitoring(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, model_registry_name, client)


if __name__=="__main__":
    main_run()