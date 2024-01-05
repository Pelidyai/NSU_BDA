from preprocessing.preprocessing import preprocess_data
from support.functions import load_x_prepared_train_data


def main_x():
    data = load_x_prepared_train_data()
    data = preprocess_data(data, skip_drop=True, skip_text_preprocessing=True,
                           skip_models_text_preprocessing=True, skip_name_desc_prediction=False,
                           skip_simple_mappings=True, skip_filling=True, skip_date_preprocess=True,
                           skip_categorical_predictions=True, skip_model_preprocess=True)
    data.to_csv('data/buf.csv', index=False)


if __name__ == '__main__':
    main_x()
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
