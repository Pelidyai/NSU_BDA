import os
import pickle

import numpy as np
from keras.losses import MAPE
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from support.constants import TARGET_NAME, FINAL_MODELS_DIR, SALARY_FROM_KEY, NAME_DESC_PREDICTION_KEY
from support.functions import load_x_prepared_train_data, smape_loss, load_y_train_norm_data, get_min_model_error


def train(x, y, save_dir, model, n=100, m=5):
    min_error = get_min_model_error(save_dir)
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        # for j in range(m):
        model.fit(x_train, y_train)
        print("iteration#", i + 1, "fit#", "___________________________")
        pred = np.asarray(model.predict(x_test)).astype('float32')
        valid_error = np.asarray(MAPE(y_test, pred)).astype('float32').mean()
        print("valid MAE:", valid_error)
        pred = np.asarray(model.predict(x_train)).astype('float32')
        error = np.asarray(MAPE(y_train, pred)).astype('float32').mean()
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
    y_data = load_y_train_norm_data()[TARGET_NAME]
    x_data = x_data.drop(['id'], axis=1)

    x_data = np.asarray(x_data).astype('float32')
    y_data = np.asarray(y_data).astype('float32')
    train(x_data, y_data, FINAL_MODELS_DIR, RandomForestRegressor(n_estimators=100, max_features=5))


if __name__ == '__main__':
    main()

