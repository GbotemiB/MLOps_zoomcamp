# get_ipython().system('pip freeze | grep scikit-learn')

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

year = int(sys.argv[1])  # year
month = int(sys.argv[2])  # month


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def calculate_average(values):
    total = sum(values)
    count = len(values)
    average = total / count
    return average


df = read_data(
    f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
)


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(calculate_average(y_pred))

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predictions': y_pred})
df_result

df_result.to_parquet(
    f'output_file_{year}_{month}.parquet',
    engine='pyarrow',
    compression=None,
    index=False,
)
