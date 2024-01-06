import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from support.constants import SALARY_FROM_RECOVER_MODELS_DIR, SALARY_FROM_KEY
from support.functions import load_x_prepared_train_data, smape_loss, get_min_model_error, normalize_column
from support.scaling import get_sf_scaler


def train(x, y, save_dir, model, n=100):
    min_error = get_min_model_error(save_dir)
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        model.fit(x_train, y_train)
        print("iteration#", i + 1, "___________________________")
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
    x_data = x_data[x_data[SALARY_FROM_KEY].notna()]
    y_data = normalize_column(x_data, SALARY_FROM_KEY, get_sf_scaler())[SALARY_FROM_KEY]

    x_data = x_data.drop([SALARY_FROM_KEY, 'id', 'created_at', 'published_at'], axis=1)
    x_data = np.asarray(x_data).astype('float32')
    y_data = np.asarray(y_data).astype('float32')
    train(x_data, y_data, SALARY_FROM_RECOVER_MODELS_DIR, RandomForestRegressor(n_estimators=100, max_features=5,
                                                                                warm_start=True))


if __name__ == '__main__':
    main()

