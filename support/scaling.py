import os.path
import pickle

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from support.constants import TARGET_NAME, SCALER_PATH, SALARY_FROM_KEY
from support.functions import load_y_train_data, load_x_train_data


def _create_scaler(data: DataFrame, target: str) -> MinMaxScaler:
    data = data.copy()
    scaler = MinMaxScaler(feature_range=(1, 3000))
    scaler.fit(np.asarray(data[target]).astype('float32').reshape(-1, 1))
    return scaler


def _reset_scaler():
    y_data = load_y_train_data()
    scaler = _create_scaler(y_data, TARGET_NAME)
    with open(SCALER_PATH + '\\scaler.pickle', 'wb') as file:
        pickle.dump(scaler, file)


def get_scaler() -> MinMaxScaler:
    if not os.path.exists(SCALER_PATH + '\\scaler.pickle'):
        _reset_scaler()
    with open(SCALER_PATH + '\\scaler.pickle', 'rb') as file:
        return pickle.load(file)


def _reset_sf_scaler():
    x_data = load_x_train_data()
    scaler = _create_scaler(x_data, SALARY_FROM_KEY)
    with open(SCALER_PATH + '\\sf_scaler.pickle', 'wb') as file:
        pickle.dump(scaler, file)


def get_sf_scaler() -> MinMaxScaler:
    if not os.path.exists(SCALER_PATH + '\\sf_scaler.pickle'):
        _reset_sf_scaler()
    with open(SCALER_PATH + '\\sf_scaler.pickle', 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    _reset_sf_scaler()
