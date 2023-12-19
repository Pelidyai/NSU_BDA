import numpy as np
import pandas as pd
from pandas import DataFrame

from learn.final_model_2 import load_model_to_tune, load_big_model
from preprocessing.preprocessing import inverse, preprocess_data
from support.constants import TARGET_NAME
from support.functions import load_x_test_data


def preprocess_x_test():
    data = load_x_test_data()
    data = preprocess_data(data)
    data.to_csv('prep_x_test7.csv', index=False)


def main():
    data = pd.read_csv('prep_x_test7.csv')
    x = np.asarray(data.drop(['id'], axis=1)).astype('float32')
    model = load_big_model()
    y = np.asarray(model.predict(x)).astype('float32')

    result = DataFrame()
    result['id'] = data['id']
    result[TARGET_NAME] = y
    result.to_csv('result7.csv', index=False)


if __name__ == '__main__':
    main()
