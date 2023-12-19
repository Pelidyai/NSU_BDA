import os
import pickle

import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from support.constants import TARGET_NAME, ENSEMBLE_MODELS_DIR
from support.functions import load_x_prepared_train_data, smape_loss, load_y_train_norm_data, get_min_model_error, \
    load_y_train_data


def train(x, y, save_dir, model, n=100, m=5):
    min_error = get_min_model_error(save_dir)
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        # for j in range(m):
        model.fit(x_train, y_train)
        print("iteration#", i + 1, "fit#", "___________________________")
        pred = np.asarray(model.predict(x_test)).astype('float32')
        valid_error = np.asarray(smape_loss(y_test, pred)).astype('float32').mean()
        print("valid MAE:", valid_error)
        pred = np.asarray(model.predict(x_train)).astype('float32')
        error = np.asarray(smape_loss(y_train, pred)).astype('float32').mean()
        print("train MAE:", error)
        print("best MAE:", min_error)
        if valid_error < min_error:
            filename = os.path.join(save_dir, 'best' + str(round(valid_error, 4)) + '.pickaim')
            pickle.dump(model, open(filename, 'wb'))
            filename = os.path.join(save_dir, 'best.pickaim')
            pickle.dump(model, open(filename, 'wb'))
            min_error = valid_error


def main():
    x_data = load_x_prepared_train_data()
    y_data = load_y_train_data()[TARGET_NAME]

    x_data = x_data.drop(['id'], axis=1)
    x_data = np.asarray(x_data).astype('float32')
    y_data = np.asarray(y_data).astype('float32')
    train(x_data, y_data, ENSEMBLE_MODELS_DIR, MLPRegressor(hidden_layer_sizes=(32, 32, 8), max_iter=1000,
                                                            early_stopping=True, warm_start=True,
                                                            learning_rate="invscaling", validation_fraction=0.1),
          n=10000)
    # MLPRegressor(hidden_layer_sizes=(32, 32, 8), max_iter=1000,
    #                                                          warm_start=True, learning_rate="invscaling"), n=1000)
    #  RandomForestRegressor(n_estimators=100) 25.51


if __name__ == '__main__':
    main()

