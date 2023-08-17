import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from prepare_features import prepare

import optuna

def preprocess(data_path):
    data = pd.read_csv(data_path)
    data = prepare(data)

    X = data.drop(columns=['price'], axis=0)
    y = data.price
    print("preprocessing completed")
    return X, y

def split_data(features, label):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    print("data splitting completed")
    return X_train, X_test, y_train, y_test

def save_train_and_test_data(X_train, X_test, y_train, y_test):
    df_train = pd.merge(X_train, y_train, left_index=True, right_index=True, how='inner')
    df_train.to_csv("./data/train_data.csv", index=False)
    
    df_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='inner')
    df_test.to_csv("./data/test_data.csv", index=False)
    return "saved train and test data successfully"

def train_model(X_train, y_train, objective):
    print("model training starting............................")
    study = optuna.create_study(direction='minimize')  # Create a new study.
    study.optimize(objective, n_trials=10)
    print("model training completed")


def objective(trial):

    with mlflow.start_run():

        max_depth = trial.suggest_int('rf_max_depth', 2, 16)
        num_leaves = trial.suggest_int('num_leaves', 2, 20)
        n_estimators = trial.suggest_int('n_estimators', 100, 4000)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
        subsample = trial.suggest_float('subsample', 0, 1)

        params = {
            'num_leaves':num_leaves,
            'max_depth':max_depth,
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
        }

        mlflow.log_params(params)

        fold_pred = []
        splits = 5
        fold = KFold(n_splits=splits)

        for data_index, test_index in fold.split(X_train, y_train):
            X_data, X_val = X_train.iloc[data_index], X_train.iloc[test_index]
            y_data, y_val = np.sqrt(y_train.iloc[data_index]), y_train.iloc[test_index]

            model = LGBMRegressor(**params, objective='rmse', force_col_wise=True, verbose=0)
            model.fit(X_data, y_data, eval_set=[(X_data, y_data), (X_val, y_val)])
            model_preds = model.predict(X_val)

            rmse = mean_squared_error(y_val, np.square(model_preds), squared=False)
            print(f'err: {rmse}')
            fold_pred.append(rmse)

        RMSE = np.mean(fold_pred)

        mlflow.log_param("splits", splits)
        mlflow.log_metric("rmse", RMSE)
        
        with open('models/lgb.bin', 'wb') as f:
            pickle.dump(model, f)

        mlflow.log_artifact(local_path="models/lgb.bin", artifact_path="models_pickle")
        
        mlflow.lightgbm.log_model(model, artifact_path="models_mlflow")

    return RMSE

    # study = optuna.create_study(direction='minimize')  # Create a new study.
    # study.optimize(objective, n_trials=10)


def best_run_and_id(EXPERIMENT_NAME, n_trials):

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

def register_model(run_id, model_registry_name):
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/models",
        name=model_registry_name
    )
    return "model registered successfully"


def transition_model(latest_version):
    client.transition_model_version_stage(
            name=model_registry_name,
            version=latest_version,
            stage="Staging",
            archive_existing_versions=False,
        )
    return "model transition successful"


if __name__=="__main__":

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = 'housing-price'
    model_registry_name = 'housing_price'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    data_path = 'data/Housing_dataset_train.csv'

    n_trials = 5

    X, y = preprocess(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    save_train_and_test_data(X_train, X_test, y_train, y_test)

    # study = optuna.create_study(direction='minimize')  # Create a new study.
    # study.optimize(objective, n_trials=n_trials)

    # best_run, run_id = best_run_and_id(EXPERIMENT_NAME, n_trials)
    # register_model(run_id, model_registry_name)

    # latest_version = client.get_latest_versions(name=model_registry_name, stages=['None'])[0].version
    # transition_model(latest_version)