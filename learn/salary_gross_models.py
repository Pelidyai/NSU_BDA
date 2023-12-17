import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from support.constants import SALARY_GROSS_RECOVER_MODELS_DIR
from support.functions import load_x_prepared_train_data

TARGET = 'salary_gross'
FEATURES = ['has_test', 'response_letter_required',
            "name_0", "name_1", "name_2", "name_3",
            "name_4", "name_5", "name_6", "name_7",
            "description_0", "description_1", "description_2", "description_3",
            "description_4", "description_5", "description_6", "description_7"]


def get_max_model_accuracy(models_dir):
    max_accuracy = 0
    for file in os.listdir(models_dir):
        file = file.replace('best', '')
        file = file.replace('.pickaim', '')
        accuracy = max_accuracy
        if file != '':
            accuracy = float(file)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
    return max_accuracy


def train(x, y, save_dir, model, n=100):
    max_accuracy = get_max_model_accuracy(save_dir)
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        model.fit(x_train, y_train)
        print("iteration#", i + 1, "___________________________")
        valid_error = accuracy_score(y_test, model.predict(x_test))
        print("valid accuracy:", valid_error)
        error = accuracy_score(y_train, model.predict(x_train))
        print("train accuracy:", error)
        print("best accuracy:", max_accuracy)
        if valid_error > max_accuracy:
            filename = os.path.join(save_dir, 'best' + str(round(valid_error, 2)) + '.pickaim')
            pickle.dump(model, open(filename, 'wb'))
            filename = os.path.join(save_dir, 'best.pickaim')
            pickle.dump(model, open(filename, 'wb'))
            max_accuracy = valid_error


def main():
    data = load_x_prepared_train_data()
    x_data = data[FEATURES].to_numpy()
    y_data = data[TARGET].to_numpy()
    train(x_data, y_data, SALARY_GROSS_RECOVER_MODELS_DIR, RandomForestRegressor(n_estimators=150))


if __name__ == '__main__':
    main()
