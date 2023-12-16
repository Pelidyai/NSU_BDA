import re

import nltk.stem
import pandas as pd
import pymorphy2
from pandas import DataFrame

from support.constants import X_TRAIN_PATH, TARGET_NAME, Y_TRAIN_PATH, PREP_X_TRAIN_PATH


def pos(word, morth=pymorphy2.MorphAnalyzer()):
    return morth.parse(word)[0].tag.POS


ROLE_TO_REMOVE = {'INTJ', 'PRCL', 'CONJ', 'PREP', 'PRED', 'ADVB', 'NUMB'}  # function words
STEMMER = nltk.stem.SnowballStemmer("russian")


def prepare_text(string: str) -> str:
    string = re.sub('<[^<]+>', "", string)
    string = re.sub("[()\"!@#$'/%\\\[\]*+.,<>^&?_=`~;:\d]", '', string)
    without_meaningless = [word for word in string.split(' ') if pos(word) not in ROLE_TO_REMOVE]
    clear = [STEMMER.stem(word.lower()) for word in without_meaningless if len(word) > 3]
    return " ".join(clear)


def load_x_train_data() -> DataFrame:
    train_data = pd.read_csv(X_TRAIN_PATH)
    return train_data


def load_x_prepared_train_data() -> DataFrame:
    train_data = pd.read_csv(PREP_X_TRAIN_PATH)
    return train_data


def load_y_train_data() -> DataFrame:
    data = pd.read_csv(Y_TRAIN_PATH)
    return data
