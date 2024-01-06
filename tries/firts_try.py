import numpy as np
import pandas as pd
from pandas import DataFrame

from models_creation import load_work_text_model
from preprocessing.preprocessing import preprocess_text, inverse, model_work
from support.constants import BERT_BASED_NAME_CHECKPOINT_DIR, TARGET_NAME
from support.functions import load_x_test_data


def preprocess_x_test():
    data = load_x_test_data()
    data = preprocess_text(data, 'name')
    data.to_csv('prep_x_test.csv', index=False)


def main():
    preprocess_x_test()
    model = load_work_text_model(BERT_BASED_NAME_CHECKPOINT_DIR)
    model.is_work = False
    data = pd.read_csv('prep_x_test.csv')
    x = np.asarray(data['name']).astype('str')
    y = model_work(model, x, 128)
    y = np.reshape(y, (1, -1))[0]
    result = DataFrame()
    result['id'] = data['id']
    result[TARGET_NAME] = inverse(y)
    result.to_csv('result1.csv', index=False)


if __name__ == '__main__':
    main()
