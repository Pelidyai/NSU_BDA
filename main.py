from preprocessing.preprocessing import preprocess_data, logo_normalize, inverse
from support.constants import PREP_X_TRAIN_PATH, TARGET_NAME, SCALER_PATH, Y_TRAIN_NORM_PATH
from support.functions import load_x_train_data, load_x_prepared_train_data, load_y_train_data


def main_x():
    data = load_x_prepared_train_data()
    data = preprocess_data(data, skip_drop=True, skip_models_text_preprocessing=True, skip_text_preprocessing=True)
    data.to_csv(PREP_X_TRAIN_PATH, index=False)


def main_y():
    data = load_y_train_data()
    norm = logo_normalize(data, TARGET_NAME)
    norm.to_csv(Y_TRAIN_NORM_PATH, index=False)


if __name__ == '__main__':
    main_x()
