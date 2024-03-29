import numpy as np
import pandas as pd
from pandas import DataFrame

from models_creation import load_name_desc_model, load_final_nn_model, load_eval_nn_model
from preprocessing.preprocessing import inverse, preprocess_data
from support.constants import TARGET_NAME, NAMES_AND_DESC_FEATURES, FINAL_MODELS_DIR, EVAL_MODELS_DIR
from support.functions import load_x_test_data


def preprocess_x_test():
    data = pd.read_csv('prep_x_test2.csv')
    try:
        data = data.drop('Unnamed: 0', axis=1)
    except Exception:
        pass
    data = preprocess_data(data, skip_drop=True, skip_text_preprocessing=True,
                           skip_models_text_preprocessing=True, skip_name_desc_prediction=False,
                           skip_simple_mappings=False, skip_filling=False, skip_date_preprocess=False,
                           skip_categorical_predictions=False, skip_second_drop=True,
                           skip_model_preprocess=True, skip_third_drop=True)
    data.to_csv('prep_x_test_eval_new.csv', index=False)


def main():
    data = pd.read_csv('prep_x_test_eval_new.csv')
    orig_data = data.copy()
    data = data.drop(['id'], axis=1)
    x = np.asarray(data).astype('float32')
    model = load_eval_nn_model(EVAL_MODELS_DIR)
    y = np.asarray(model.predict(x)).astype('float32')

    result = DataFrame()
    result['id'] = orig_data['id']
    result[TARGET_NAME] = inverse(y)
    result.to_csv('result_final_eval_new.csv', index=False)


if __name__ == '__main__':
    main()
