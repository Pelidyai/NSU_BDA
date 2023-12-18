import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame

from learn.final_model_2 import load_model_to_tune
from preprocessing.preprocessing import inverse, preprocess_data
from support.constants import TARGET_NAME
from support.functions import load_x_test_data


def preprocess_x_test():
    data = load_x_test_data()
    data = preprocess_data(data)
    data.to_csv('prep_x_test3.csv', index=False)


def main():
    data = pd.read_csv('prep_x_test3.csv')
    x = np.asarray(data.drop(['id'], axis=1)).astype('float32')
    with open('../models/final_forest/best.pickaim', 'rb') as file:
        model = pickle.load(file)
    y = np.asarray(model.predict(x)).astype('float32')

    result = DataFrame()
    result['id'] = data['id']
    result[TARGET_NAME] = inverse(y)
    result.to_csv('result5.csv', index=False)


if __name__ == '__main__':
    main()
