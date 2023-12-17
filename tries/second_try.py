import numpy as np
import pandas as pd
from pandas import DataFrame

from models_creation import load_work_text_model
from preprocessing.preprocessing import preprocess_text, inverse, model_work, preprocess_data
from support.constants import BERT_BASED_NAME_CHECKPOINT_DIR, TARGET_NAME, PREP_X_TEST_PATH
from support.functions import load_x_test_data


def preprocess_x_test():
    data = load_x_test_data()
    data = preprocess_data(data)
    data.to_csv(PREP_X_TEST_PATH)


def main():
    model = load_work_text_model(BERT_BASED_NAME_CHECKPOINT_DIR)
    data = pd.read_csv(PREP_X_TEST_PATH)
    x = np.asarray(data['name']).astype('str')
    y = model_work(model, x, 128)
    result = DataFrame()
    result['id'] = data['id']
    result[TARGET_NAME] = inverse(y)
    result.to_csv('result.csv', index=False)


if __name__ == '__main__':
    preprocess_x_test()
