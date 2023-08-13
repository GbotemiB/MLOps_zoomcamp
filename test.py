import requests
from prepare_features import prepare

housing_details = {
    "ID": 323,
    "loc": "lagos",
    "title": "Mansion",
    "bedroom": 2,
    "bathroom": 1,
    "parking_space" : 2
}

url = 'http://127.0.0.1:9090/predict'
response = requests.post(url, json=housing_details)
# features = prepare(response)
print(response.json())