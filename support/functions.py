import math
import os
import re
from typing import Any, Iterable

import nltk.stem
import numpy as np
import pandas as pd
import pymorphy2
import tensorflow as tf
import tensorflow.python.keras.backend as K
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from support.constants import X_TRAIN_PATH, Y_TRAIN_PATH, PREP_X_TRAIN_PATH, Y_TRAIN_NORM_PATH, \
    X_TEST_PATH, PREP_X_TEST_PATH, ENDPOINT_X_SCALE


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


def load_x_train_data() -> DataFrame:
    train_data = pd.read_csv(X_TRAIN_PATH)
    return train_data


def load_x_prepared_train_data() -> DataFrame:
    train_data = pd.read_csv(PREP_X_TRAIN_PATH)
    return train_data


def load_x_prepared_test_data() -> DataFrame:
    train_data = pd.read_csv(PREP_X_TEST_PATH)
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


def get_min_model_error(models_dir):
    min_error = 10000
    for file in os.listdir(models_dir):
        try:
            file.index('.pickaim')
        except ValueError:
            continue
        file = file.replace('best', '')
        file = file.replace('.pickaim', '')
        error = min_error
        if file != '':
            error = float(file)
        if error < min_error:
            min_error = error
    return min_error


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.05)


def normalize(y_to_norm: Iterable, scaler: MinMaxScaler) -> Iterable:
    transformed = scaler.transform(np.asarray(y_to_norm).astype('float32').reshape(-1, 1))
    result = np.asarray(list(map(lambda x: (math.log(x) + 1) * ENDPOINT_X_SCALE, transformed))).astype('float32')
    return result


def normalize_column(data: DataFrame, target: str, scaler: MinMaxScaler) -> DataFrame:
    normalized = data.copy()
    to_transform = np.asarray(data[target]).astype('float32')
    normalized[target] = normalize(to_transform, scaler)
    return normalized
