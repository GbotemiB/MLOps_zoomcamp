import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from prefect import task, flow

@task()
def best_performing_model(client, experiment):

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )[0]

    best_run_id = runs.info.run_id
    best_run_metric = runs.data.metrics["rmse"]

    for version in client.search_model_versions():
        if version.run_id == best_run_id:
            best_version = version.version

    return best_run_id, best_run_metric, best_version

@task()
def production_model(client, model_registry_name):

    if client.get_latest_versions == True:
        prod_model = client.get_latest_versions(model_registry_name, stages=["Production"])[0]
        prod_model_run_id = prod_model.run_id   

        prod_model_metric = client.get_run(run_id=prod_model_run_id).data.metrics["rmse"]
    else:
        prod_model_run_id = 0
        prod_model_metric = 0
    
    return prod_model_run_id, prod_model_metric

@flow()
def runner(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, model_registry_name, client, experiment):

    best_run_id, best_run_metric, best_version = best_performing_model(client, experiment)
    prod_model_run_id, prod_model_metric = production_model(client, model_registry_name)

    if best_run_metric > prod_model_metric:
        client.transition_model_version_stage(
            name=model_registry_name,
            version=best_version,
            stage="Production",
            archive_existing_versions=True,
        )

        print("new model in production")
    else:
        print("old model is better than new model")

if __name__=="__main__":

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = 'housing-price'
    model_registry_name = "housing_price"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runner(MLFLOW_TRACKING_URI, EXPERIMENT_NAME, model_registry_name, client, experiment)
    