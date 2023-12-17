import numpy as np
import pandas as pd
from pandas import DataFrame

from learn.name_description_models import NAMES_AND_DESC_FEATURES
from models_creation import load_name_desc_model
from preprocessing.preprocessing import inverse, preprocess_data
from support.constants import TARGET_NAME
from support.functions import load_x_test_data


def preprocess_x_test():
    data = load_x_test_data()
    data = preprocess_data(data)
    data.to_csv('prep_x_test2.csv')


def main():
    data = pd.read_csv('prep_x_test2.csv')
    x = np.asarray(data[NAMES_AND_DESC_FEATURES]).astype('float32')
    model = load_name_desc_model()
    y = np.asarray(model.predict(x)).astype('float32')

    result = DataFrame()
    result['id'] = data['id']
    result[TARGET_NAME] = inverse(y)
    result.to_csv('result2.csv', index=False)


if __name__ == '__main__':
    main()
