from preprocessing.preprocessing import preprocess_data
from support.constants import PREP_X_TRAIN_PATH
from support.functions import load_x_train_data, load_x_prepared_train_data


def main():
    data = load_x_prepared_train_data()
    data = preprocess_data(data, skip_drop=True, skip_text_preprocessing=True, skip_models_text_preprocessing=True)
    data.to_csv(PREP_X_TRAIN_PATH)


if __name__ == '__main__':
    main()
