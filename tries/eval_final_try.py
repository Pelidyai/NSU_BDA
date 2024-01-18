import numpy as np
import pandas as pd
from pandas import DataFrame

from models_creation import load_name_desc_model, load_final_nn_model, load_final_model, load_eval_final_model, \
    load_eval_final_nn_model
from preprocessing.preprocessing import inverse, preprocess_data
from support.constants import TARGET_NAME, NAMES_AND_DESC_FEATURES, FINAL_MODELS_DIR, EVAL_FINAL_MODELS_DIR
from support.functions import load_x_test_data
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_x_test():
    data = pd.read_csv('prep_x_test_eval_new.csv')
    try:
        data = data.drop('Unnamed: 0', axis=1)
    except Exception:
        pass
    data = preprocess_data(data, skip_drop=True, skip_text_preprocessing=True,
                           skip_models_text_preprocessing=True, skip_name_desc_prediction=True,
                           skip_simple_mappings=True, skip_filling=True, skip_date_preprocess=True,
                           skip_categorical_predictions=True, skip_second_drop=False, skip_model_preprocess=False,
                           skip_third_drop=False, skip_final=False)
    data.to_csv('prep_x_test_eval_final_new.csv', index=False)


def main():
    data = pd.read_csv('prep_x_test_eval_final_new.csv')
    original_data = data.copy()
    data = data.drop(['id'], axis=1)
    x = np.asarray(data).astype('float32')
    model = load_eval_final_nn_model(EVAL_FINAL_MODELS_DIR)
    y = np.asarray(model.predict(x)).astype('float32')

    result = DataFrame()
    result['id'] = original_data['id']
    result[TARGET_NAME] = inverse(y)
    result.to_csv('result_eval_final_new.csv', index=False)


if __name__ == '__main__':
    main()
