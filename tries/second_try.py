import numpy as np
import pandas as pd
from pandas import DataFrame

from models_creation import load_name_desc_model
from preprocessing.preprocessing import inverse, preprocess_data
from support.constants import TARGET_NAME, NAMES_AND_DESC_FEATURES
from support.functions import load_x_test_data


def preprocess_x_test():
    data = pd.read_csv('prep_x_test1.csv')
    data = preprocess_data(data,
                           skip_drop=True,
                           skip_text_preprocessing=True,
                           skip_models_text_preprocessing=False,
                           skip_name_desc_prediction=True,
                           skip_simple_mappings=True,
                           skip_filling=True,
                           skip_date_preprocess=True,
                           skip_categorical_predictions=True,
                           skip_model_preprocess=True)
    data.to_csv('prep_x_test2222.csv', index=False)


def main():
    data = pd.read_csv('prep_x_test2222.csv')
    x = np.asarray(data[NAMES_AND_DESC_FEATURES]).astype('float32')
    model = load_name_desc_model()
    y = np.asarray(model.predict(x)).astype('float32')

    result = DataFrame()
    result['id'] = data['id']
    result[TARGET_NAME] = y
    result.to_csv('result_name_desc_non_norm.csv', index=False)


if __name__ == '__main__':
    main()
