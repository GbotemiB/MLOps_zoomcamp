import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction import DictVectorizer

import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from prepare_features import prepare

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = 'housing-price'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

data_path = 'data/Housing_dataset_train.csv'

def preprocess(data_path):
    data = pd.read_csv(data_path)

    print(data.columns.tolist())
    
    data = prepare(data)
    print("_____________________________________________________________________________")
    print(data.head())

    X = data.drop(columns=['price'], axis=0)
    y = data.price

    return X, y

def split_data(features, label):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):


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
