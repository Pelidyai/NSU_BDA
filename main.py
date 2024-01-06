import math

import numpy as np
import pandas as pd

from preprocessing.preprocessing import preprocess_data, round_sum, inverse
from support.constants import PREP_X_TRAIN_PATH, TARGET_NAME, Y_TRAIN_NORM_PATH
from support.functions import load_x_prepared_train_data, load_x_train_data, load_y_train_data, normalize
from support.scaling import get_scaler


def main_x():
    data = load_x_prepared_train_data()
    try:
        data = data.drop('Unnamed: 0', axis=1)
    except Exception:
        pass
    data = preprocess_data(data, skip_drop=True, skip_text_preprocessing=True,
                           skip_models_text_preprocessing=True, skip_name_desc_prediction=True,
                           skip_simple_mappings=True, skip_filling=True, skip_date_preprocess=True,
                           skip_categorical_predictions=False, skip_second_drop=True, skip_model_preprocess=True)
    data.to_csv('data/buf.csv', index=False)


def main_y():
    y_data = load_y_train_data()
    to_transform = np.asarray(y_data[TARGET_NAME]).astype('float32')
    norm = normalize(to_transform, get_scaler())
    y_data[TARGET_NAME] = norm
    y_data.to_csv(Y_TRAIN_NORM_PATH, index=False)


if __name__ == '__main__':
    main_x()
    # data = pd.read_csv('tries/result_final_5k.csv')
    #
    # data[TARGET_NAME] = round_sum(np.asarray(data[TARGET_NAME]).astype('float32'))
    # data.to_csv('tries/result_final_5k_cut2.csv', index=False)

    # x_data = load_x_train_data()
    # # x_data = preprocess_date(x_data, 'published_at')
    # #
    # # x_data = None
    # file = 'data/prep_X_train_5.csv'
    # x_prep = pd.read_csv(file)
    # x_prep['published_at'] = x_data['created_at']
    # x_prep['created_at'] = x_data['created_at']
    # # x_data = load_x_train_data()
    # # x_prep['employer_name'] = preprocess_employer_name(x_data)['employer_name']
    # x_prep.to_csv(file, index=False)

    # x_data = load_x_prepared_train_data()
    # description = x_data['description']
    # lengths = description.apply(lambda x: len(x))
    # x_data['desc_length'] = lengths.apply(lambda x: math.log(x))
    # x_data.to_csv('data/prep_X_train_1_1.csv', index=False)
