import os
import mlflow
import pickle
import pandas as pd

from flask import Flask, jsonify, request
from prepare_features import prepare


with open('models/cat.bin', 'rb') as f:
    model = pickle.load(f)  


app = Flask('Housing Price Prediction')

@app.route('/', methods=['GET'])
def index():
    return "Hi, Welcome to Nigerian Housing Prediction Portal"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    parameters = request.get_json()
    data = pd.DataFrame([parameters])
    data = prepare(data)
    prediction = model.predict(data)
    result = {
        'price' : prediction[0]
    }

    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9090)