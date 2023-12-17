import re
from typing import Any

import nltk.stem
import pandas as pd
import pymorphy2
from pandas import DataFrame
import tensorflow.python.keras.backend as K

from support.constants import X_TRAIN_PATH, Y_TRAIN_PATH, PREP_X_TRAIN_PATH, Y_TRAIN_NORM_PATH, \
    X_TEST_PATH, PREP_X_TEST_PATH


def smape_loss(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape


def pos(word, morth=pymorphy2.MorphAnalyzer()):
    return morth.parse(word)[0].tag.POS


ROLE_TO_REMOVE = {'INTJ', 'ADJS', 'ADJF', 'PRTF', 'PRTS',
                  'GRND', 'COMP', 'PRCL', 'CONJ', 'PREP', 'PRED', 'ADVB', 'NUMB'}  # function words
STEMMER = nltk.stem.SnowballStemmer("russian")


def prepare_text(string: str) -> str:
    string = re.sub('<[^<]+>', "", string)
    string = re.sub("[()\"!@#$'/%\\\[\]*+.,<>^&?_=`~;:\d]", ' ', string)
    string = re.sub(" +", ' ', string)
    without_meaningless = [word.lower().replace('ё', 'е').strip()
                           for word in string.split(' ') if pos(word) not in ROLE_TO_REMOVE]
    clear = [STEMMER.stem(word) for word in without_meaningless if len(word) > 3]
    return " ".join(clear)


def load_x_test_data() -> DataFrame:
    test_data = pd.read_csv(X_TEST_PATH)
    return test_data


def load_x_prepared_test_data() -> DataFrame:
    test_data = pd.read_csv(PREP_X_TEST_PATH)
    return test_data


def load_x_train_data() -> DataFrame:
    train_data = pd.read_csv(X_TRAIN_PATH)
    return train_data


def load_x_prepared_train_data() -> DataFrame:
    train_data = pd.read_csv(PREP_X_TRAIN_PATH)
    return train_data


def load_y_train_data() -> DataFrame:
    data = pd.read_csv(Y_TRAIN_PATH)
    return data


def load_y_train_norm_data() -> DataFrame:
    data = pd.read_csv(Y_TRAIN_NORM_PATH)
    return data


def split_to_batches(any_list: list[Any], batch_size: int) -> list[Any]:
    return list(__divide_chunks(any_list, batch_size))


def __divide_chunks(any_list: list[Any], chunk_size: int):
    for i in range(0, len(any_list), chunk_size):
        yield any_list[i: i + chunk_size]
