import requests

housing_details = {
    "ID": 323,
    "loc": "lagos",
    "title": "Mansion",
    "bedroom": 2.0,
    "bathroom": 1.0,
    "parking_space": 2.0,
}

url = 'http://127.0.0.1:9090/predict'
response = requests.post(url, json=housing_details)
print(response.json())
