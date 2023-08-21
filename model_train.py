import pickle

import numpy as np
import mlflow
import optuna
import pandas as pd
from prefect import flow, task
from lightgbm import LGBMRegressor
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from prepare_features import prepare


@task()
def preprocess(data_path):
    data = pd.read_csv(data_path)
    data = prepare(data)

    X = data.drop(columns=['price'], axis=0)
    y = data.price
    print("preprocessing completed")
    return X, y


@task()
def split_data(features, label):
    X_train, X_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=42
    )
    print("data splitting completed")
    return X_train, X_test, y_train, y_test


@task()
def save_train_and_test_data(X_train, X_test, y_train, y_test):
    df_train = pd.merge(
        X_train, y_train, left_index=True, right_index=True, how='inner'
    )
    df_train.to_csv("./data/train_data.csv", index=False)

    df_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='inner')
    df_test.to_csv("./data/test_data.csv", index=False)
    return "saved train and test data successfully"


@task()
def best_run_and_id(EXPERIMENT_NAME, n_trials, client):

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    best_run = float('inf')
    run_id = None

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=n_trials,
        # order_by=["metrics.rmse"]
    )

    for run in runs:
        print(run.data.metrics['rmse'])
        # print(run.info.run_id, run.data.metrics["rmse"])
        if run.data.metrics['rmse'] < best_run:
            best_run = run.data.metrics['rmse']
            run_id = run.info.run_id
    return best_run, run_id


@task()
def register_model(run_id, model_registry_name):
    mlflow.register_model(model_uri=f"runs:/{run_id}/models", name=model_registry_name)
    return "model registered successfully"


@task()
def transition_model(latest_version, client, model_registry_name):
    client.transition_model_version_stage(
        name=model_registry_name,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=False,
    )
    return "model transition successful"


@task()
def optuna_(X_train, y_train, n_trials):
    def objective(trial):

        with mlflow.start_run():

            max_depth = trial.suggest_int('rf_max_depth', 2, 16)
            n_estimators = trial.suggest_int('n_estimators', 100, 4000)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
            subsample = trial.suggest_float('subsample', 0, 1)
            num_leaves = trial.suggest_int('num_leaves', 2, 15)

            params = {
                'max_depth': max_depth,
                'colsample_bytree': colsample_bytree,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'subsample': subsample,
                'num_leaves': num_leaves,
            }

            mlflow.log_params(params)

            fold_pred = []
            splits = 5
            fold = KFold(n_splits=splits)

            for data_index, test_index in fold.split(X_train, y_train):
                X_data, X_test = X_train.iloc[data_index], X_train.iloc[test_index]
                y_data, y_test = (
                    np.sqrt(y_train.iloc[data_index]),
                    y_train.iloc[test_index],
                )

                model = LGBMRegressor(**params, objective='rmse')
                model.fit(X_data, y_data, eval_set=[(X_data, y_data), (X_test, y_test)])
                model_preds = model.predict(X_test)

                rmse = mean_squared_error(y_test, np.square(model_preds), squared=False)
                print(f'err: {rmse}')
                fold_pred.append(rmse)

            RMSE = np.mean(fold_pred)

            mlflow.log_param("splits", splits)
            mlflow.log_metric("rmse", RMSE)

            with open('models/lgb.bin', 'wb') as f:
                pickle.dump(model, f)

            mlflow.log_artifact(
                local_path="models/lgb.bin", artifact_path="models_pickle"
            )
            mlflow.lightgbm.log_model(model, artifact_path="models_mlflow")

        return RMSE  # An objective value linked with the Trial object.

    # train_model(X_train, y_train, objective)

    study = optuna.create_study(direction='minimize')  # Create a new study.
    study.optimize(objective, n_trials=n_trials)


@flow()
def run(
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    model_registry_name,
    client,
    data_path,
    n_trials,
):

    X, y = preprocess(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    save_train_and_test_data(X_train, X_test, y_train, y_test)

    optuna_(X_train, y_train, n_trials)

    best_run, run_id = best_run_and_id(EXPERIMENT_NAME, n_trials, client)
    register_model(run_id, model_registry_name)

    latest_version = client.get_latest_versions(
        name=model_registry_name, stages=['None']
    )[0].version
    transition_model(latest_version, client, model_registry_name=model_registry_name)


if __name__ == "__main__":
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = 'housing-price'
    model_registry_name = 'housing_price'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = 'data/Housing_dataset_train.csv'
    n_trials = 2

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    run(
        MLFLOW_TRACKING_URI,
        EXPERIMENT_NAME,
        model_registry_name,
        client,
        data_path,
        n_trials,
    )
