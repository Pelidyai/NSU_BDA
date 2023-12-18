import pandas as pd

from preprocessing.preprocessing import preprocess_data, logo_normalize, inverse, preprocess_area_name, \
    preprocess_employer_name
from support.constants import PREP_X_TRAIN_PATH, TARGET_NAME, SCALER_PATH, Y_TRAIN_NORM_PATH
from support.functions import load_x_train_data, load_x_prepared_train_data, load_y_train_data


def main_x():
    data = load_x_prepared_train_data()
    data = preprocess_data(data, skip_drop=True, skip_text_preprocessing=True,
                           skip_models_text_preprocessing=True, skip_name_desc_prediction=True,
                           skip_simple_mappings=True, skip_filling=False)
    data.to_csv(PREP_X_TRAIN_PATH, index=False)


def main_y():
    data = load_y_train_data()
    norm = logo_normalize(data, TARGET_NAME)
    norm.to_csv(Y_TRAIN_NORM_PATH, index=False)


if __name__ == '__main__':
    main_x()
    # file = 'data/prep_X_train.csv'
    # x_prep = pd.read_csv(file)
    # x_prep = x_prep.drop(["salary_gross"], axis=1)
    # # x_data = load_x_train_data()
    # # x_prep['employer_name'] = preprocess_employer_name(x_data)['employer_name']
    # x_prep.to_csv(file, index=False)
