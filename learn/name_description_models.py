import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from preprocessing.preprocessing import inverse
from support.constants import TARGET_NAME, NAMES_AND_DESC_FEATURES, RU_NAME_DESC_MODELS_DIR
from support.functions import load_x_prepared_train_data, smape_loss, load_y_train_norm_data, load_y_train_data, \
    get_min_model_error


def train(x, y, save_dir, model, n=100):
    min_error = get_min_model_error(save_dir)
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        model.fit(x_train, y_train)
        print("iteration#", i + 1, "___________________________")
        pred = np.asarray(model.predict(x_test)).astype('float32')
        valid_error = np.asarray(tf.keras.losses.MAPE(y_test, pred)).astype('float32').mean()
        print("valid MAE:", valid_error)
        pred = np.asarray(model.predict(x_train)).astype('float32')
        error = np.asarray(tf.keras.losses.MAPE(y_train, pred)).astype('float32').mean()
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
    y_data = load_y_train_norm_data()

    x_data = np.asarray(x_data[NAMES_AND_DESC_FEATURES]).astype('float32')
    y_data = np.asarray(y_data[TARGET_NAME]).astype('float32')
    train(x_data, y_data, RU_NAME_DESC_MODELS_DIR, RandomForestRegressor(n_estimators=100, max_features=5))
    # train(x_data, y_data, NAME_DESC_MODELS_DIR, GradientBoostingRegressor(verbose=1))
    # train(x_data, y_data, NAME_DESC_MODELS_DIR, MLPRegressor(hidden_layer_sizes=(64, 32, 8), activation='identity'))


def main_test():
    x_data = load_x_prepared_train_data()
    x_data = np.asarray(x_data[NAMES_AND_DESC_FEATURES]).astype('float32')

    y_data = load_y_train_data()
    y_data = np.asarray(y_data[TARGET_NAME]).astype('float32')

    y_data_norm = load_y_train_norm_data()
    y_data_norm = np.asarray(y_data_norm[TARGET_NAME]).astype('float32')

    with open(RU_NAME_DESC_MODELS_DIR + "/best.pickaim", 'rb') as file:
        model = pickle.load(file)
        result_norm = np.asarray(model.predict(x_data)).astype('float32')
        error_norm = smape_loss(y_data_norm, result_norm).numpy().mean()
        result = inverse(result_norm)
        error = smape_loss(y_data, result).numpy().mean()
        print(error_norm, error)


if __name__ == '__main__':
    main()

