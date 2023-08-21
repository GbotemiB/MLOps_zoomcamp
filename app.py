import mlflow
import pandas as pd
from flask import Flask, jsonify, request
from mlflow import MlflowClient

from prepare_features import prepare

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = 'housing-price'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

model_registry_name = "housing_price"

client = MlflowClient()
prod_model = client.get_latest_versions(model_registry_name, stages=["Production"])[0]
prod_model_run_id = prod_model.run_id
logged_model = f'runs:/{prod_model_run_id}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

app = Flask('Housing Price Prediction')


@app.route('/', methods=['GET'])
def index():
    return "Hi, Welcome to Nigerian Housing Prediction Portal"


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    parameters = request.get_json()
    data = pd.DataFrame([parameters])
    data = prepare(data)
    prediction = loaded_model.predict(data)
    result = {'price': float(prediction[0])}

    return jsonify(result)


if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=9090)
