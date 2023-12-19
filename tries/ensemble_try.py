import numpy as np
import pandas as pd
from pandas import DataFrame

from learn.final_model_2 import load_model_to_tune, load_big_model, load_final_model
from models_creation import load_final_simple_model, load_ensemble_model
from preprocessing.preprocessing import inverse, preprocess_data, preprocess_with_models
from support.constants import TARGET_NAME
from support.functions import load_x_test_data


def preprocess_x_test():
    data = load_x_test_data()
    data = preprocess_data(data)
    data.to_csv('prep_x_test_ensemble2.csv', index=False)


def main():
    data = pd.read_csv('prep_x_test_ensemble2.csv')
    x = np.asarray(data.drop(['id'], axis=1)).astype('float32')
    model = load_ensemble_model()
    y = np.asarray(model.predict(x)).astype('float32')

    result = DataFrame()
    result['id'] = data['id']
    result[TARGET_NAME] = y
    result.to_csv('result_ensemble3.csv', index=False)


if __name__ == '__main__':
    main()
