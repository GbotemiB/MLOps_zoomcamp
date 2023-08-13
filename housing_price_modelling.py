import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction import DictVectorizer
import optuna

import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = 'housing-price'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

for run in runs:
    print(f'run_id:{run.info.run_id}, rmse:{run.data.metrics["rmse"]}')

run_id = "332da12b29be4a7fb4a05ce3e9e9d5ff"
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri=model_uri, name="housing_price")

model_name = "housing_price"
latest_version = client.get_latest_versions(name=model_name)


latest_version[0].version

client.transition_model_version_stage(
    name=model_name,
    version=latest_version[0].version,
    stage="Staging",
    archive_existing_versions=False
)

data_path = 'data/Housing_dataset_train.csv'

state_to_zone = {
    "Abia": "South-East",
    "Adamawa": "North-East",
    "Akwa Ibom": "South-South",
    "Anambra": "South-East",
    "Bauchi": "North-East",
    "Bayelsa": "South-South",
    "Benue": "North-Central",
    "Borno": "North-East",
    "Cross River": "South-South",
    "Delta": "South-South",
    "Ebonyi": "South-East",
    "Edo": "South-South",
    "Ekiti": "South-West",
    "Enugu": "South-East",
    "Gombe": "North-East",
    "Imo": "South-East",
    "Jigawa": "North-West",
    "Kaduna": "North-West",
    "Kano": "North-West",
    "Katsina": "North-West",
    "Kebbi": "North-West",
    "Kogi": "North-Central",
    "Kwara": "North-Central",
    "Lagos": "South-West",
    "Nasarawa": "North-Central",
    "Niger": "North-Central",
    "Ogun": "South-West",
    "Ondo": "South-West",
    "Osun": "South-West",
    "Oyo": "South-West",
    "Plateau": "North-Central",
    "Rivers": "South-South",
    "Sokoto": "North-West",
    "Taraba": "North-East",
    "Yobe": "North-East",
    "Zamfara": "North-West",
}


house_type_ranks = {
    'Cottage': 1,
    'Bungalow': 2,
    'Townhouse': 3,
    'Terrace duplex': 4,
    'Detached duplex': 5,
    'Semi-detached duplex': 6,
    'Flat': 7,
    'Penthouse': 8,
    'Apartment': 9,
    'Mansion': 10
}

def preprocess(data_path):
    data = pd.read_csv(data_path)

    print(data.columns.tolist())
    
    data['zone'] = data['loc'].map(state_to_zone)
    data['title'] = data['title'].map(house_type_ranks)

    category_frequencies = data['loc'].value_counts(normalize=True)
    loc_frequency_mapping = category_frequencies.to_dict()
    data['loc'] = data['loc'].map(loc_frequency_mapping)

    data['rooms'] = data['bathroom'] + data['bedroom']
    data['bathroom_ratio'] = data['bathroom']/(data['bathroom'] + data['bedroom'])

    data['zone'] = data['zone'].astype('category').cat.codes

    print("_____________________________________________________________________________")
    print(data.head())

    X = data.drop(columns=['price'], axis=0)
    y = data.price

    return X, y


X_, y_ = preprocess(data_path)


mlflow.lightgbm.autolog(disable=True)

with mlflow.start_run():

    params = {
        'max_depth': 10,
        'n_estimators': 2000,
        'learning_rate': 0.002712819361612371,
        'colsample_bytree': 0.9484547548287134,
        'subsample': 0.8490126211976283
        }

    mlflow.log_params(params)

    fold_pred = []
    splits = 2
    fold = KFold(n_splits=splits)

    for data_index, test_index in fold.split(X_, y_):
        X_data, X_test = X_.iloc[data_index], X_.iloc[test_index]
        y_data, y_test = np.sqrt(y_.iloc[data_index]), y_.iloc[test_index]

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

    mlflow.log_artifact(local_path="models/lgb.bin", artifact_path="models_pickle")
    mlflow.lightgbm.log_model(model, artifact_path="models_mlflow")


with mlflow.start_run():

    params = {
        'max_depth': 10,
        'n_estimators': 2000,
        'subsample': 0.84,
        'learning_rate': 0.01,
        'n_estimators' : 2000
        }

    mlflow.log_params(params)

    fold_pred_1 = []
    splits = 10
    fold = KFold(n_splits=splits)

    for data_index, test_index in fold.split(X_, y_):
        X_data, X_test = X_.iloc[data_index], X_.iloc[test_index]
        y_data, y_test = np.sqrt(y_.iloc[data_index]), y_.iloc[test_index]

        model = CatBoostRegressor(**params)
        model.fit(X_data, y_data, eval_set=[(X_data, y_data), (X_test, y_test)], verbose=0)
        model_preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, np.square(model_preds), squared=False)
        print(f'err: {rmse}')
        fold_pred_1.append(rmse)

    RMSE = np.mean(fold_pred_1)

    mlflow.log_param("splits", splits)
    mlflow.log_metric("rmse", RMSE)
    
    with open('models/cat.bin', 'wb') as f:
        pickle.dump(model, f)

    mlflow.log_artifact(local_path="models/cat.bin", artifact_path="models_pickle")
    mlflow.catboost.log_model(model, artifact_path="models_mlflow")

with mlflow.start_run():

    params = {
        'max_depth': 10,
        'n_estimators': 2000,
        'subsample': 0.84,
        'learning_rate': 0.01,
        'n_estimators' : 2000
        }

    mlflow.log_params(params)

    fold_pred_1 = []
    splits = 10
    fold = KFold(n_splits=splits)

    for data_index, test_index in fold.split(X_, y_):
        X_data, X_test = X_.iloc[data_index], X_.iloc[test_index]
        y_data, y_test = np.sqrt(y_.iloc[data_index]), y_.iloc[test_index]

        model = XGBRegressor(**params)
        model.fit(X_data, y_data, eval_set=[(X_data, y_data), (X_test, y_test)], verbose=0)
        model_preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, np.square(model_preds), squared=False)
        print(f'err: {rmse}')
        fold_pred_1.append(rmse)

    RMSE = np.mean(fold_pred_1)

    mlflow.log_param("splits", splits)
    mlflow.log_metric("rmse", RMSE)
    
    with open('models/xgb.bin', 'wb') as f:
        pickle.dump(model, f)

    mlflow.log_artifact(local_path="models/xgb.bin", artifact_path="models_pickle")
    mlflow.xgboost.log_model(model, artifact_path="models_mlflow")






# In[ ]:


# def objective(trial):

#     max_depth = trial.suggest_int('rf_max_depth', 2, 32)
#     n_estimators = trial.suggest_int('n_estimators', 100, 4000)
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
#     colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
#     subsample = trial.suggest_float('subsample', 0, 1)

#     params = {
#         'max_depth':max_depth,
#         'colsample_bytree': colsample_bytree,
#         'learning_rate': learning_rate,
#         'n_estimators': n_estimators,
#         'subsample': subsample,
#     }

#     X_data, X_val, y_data, y_val = data_test_split(X, y, random_state=RANDOM_STATE)

#     LGB = CatBoostRegressor(**params)
#     LGB.fit(X_data, y_data)
#     y_pred = LGB.predict(X_val)

#     error = mean_squared_error(y_val, y_pred, squared=False)

#     return error  # An objective value linked with the Trial object.

#  # Invoke optimization of the objective function.


# In[ ]:


def objective(trial):

    max_depth = trial.suggest_int('rf_max_depth', 2, 16)
    n_estimators = trial.suggest_int('n_estimators', 100, 4000)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
#     colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
    subsample = trial.suggest_float('subsample', 0, 1)

    params = {
        'max_depth':max_depth,
#         'colsample_bytree': colsample_bytree,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'subsample': subsample,
    }

    X_data, X_val, y_data, y_val = data_test_split(X, y, random_state=RANDOM_STATE)

    CAT = CatBoostRegressor(**params)
    CAT.fit(X_data, y_data, verbose=0)
    y_pred = CAT.predict(X_val)

    error = mean_squared_error(y_val, y_pred, squared=False)

    return error  # An objective value linked with the Trial object.

study = optuna.create_study(direction='minimize')  # Create a new study.
study.optimize(objective, n_trials=100)
study.best_trial
trial = study.best_trial
trial.value
trial.params
