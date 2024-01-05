import math
import pickle
from datetime import datetime
from typing import Iterable

import numpy as np
from pandas import DataFrame

from learn.rubert_emb_based import load_work_rubert_text_model
from learn.salary_from_models2 import create_data_to_eval_salary_from, load_salary_from_nn_model
from models_creation import load_work_text_model, load_name_desc_model, load_salary_from_model, load_categorical_model, \
    RuBert, load_name_desc_nn_model, load_categorical_nn_model, load_eval_nn_model, load_eval_model
from support.constants import BERT_BASED_NAME_CHECKPOINT_DIR, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR, \
    NAME_DESC_PREDICTION_KEY, NAMES_AND_DESC_FEATURES, SALARY_FROM_KEY, CATEGORICAL_KEY, MLP_MODEL_PATH, \
    GRAD_MODEL_PATH, RFR_MODEL_PATH, RU_BERT_BASED_NAME_CHECKPOINT_DIR, RU_BERT_BASED_DESCRIPTION_CHECKPOINT_DIR, \
    RU_NAME_DESC_MODELS_DIR, CATEGORICAL_FEATURES, CATEGORICAL_DIR, EVAL_MODELS_DIR, EVAL_SALARY_FROM_RECOVER_MODELS_DIR
from support.functions import prepare_text, split_to_batches


def preprocess_text(data: DataFrame, key: str) -> DataFrame:
    key_frame = data[key]
    key_frame = key_frame.fillna('empty')
    print(f'Start preprocess key - {key}')
    prepared_frame = key_frame.apply(prepare_text)
    print(f'End preprocess key - {key}')
    prepared_frame = prepared_frame.fillna('empty')
    data[key] = prepared_frame
    return data


def cut(arrays, idx) -> list:
    result = []
    for array in arrays:
        result.append(array[idx])
    return result


def model_work(model, input_texts, batch_size) -> list[np.array]:
    output = []
    batches = split_to_batches(input_texts, batch_size)
    i = 0
    for batch in batches:
        output.extend(model(batch).numpy())
        i += 1
        print(f'Model preprocess {i}/{len(batches)} batches')
    return output


def preprocess_text_with_model(data: DataFrame, checkpoint_dir: str, x_key: str) -> DataFrame:
    loaded_model = load_work_text_model(checkpoint_dir)
    input_texts = np.asarray(data[x_key]).astype('str')
    batch_size = 128
    print(f'Start model preprocess text key - {x_key}')
    output = model_work(loaded_model, input_texts, batch_size)
    data = data.drop([x_key], axis=1)
    for i in range(len(output[0])):
        data[f'{x_key}_{i}'] = cut(output, i)
    print(f'End model preprocess text key - {x_key}')
    return data


def preprocess_text_with_rubert(data: DataFrame, checkpoint_dir: str, x_key: str) -> DataFrame:
    model = RuBert()
    rubert_encode_model = load_work_rubert_text_model(checkpoint_dir)
    input_texts = np.asarray(data[x_key]).astype('str').tolist()
    batch_size = 128
    output = model_work(model.embed, input_texts, batch_size)
    output = np.asarray(output).astype('float64')
    output = model_work(rubert_encode_model, output, batch_size)
    data = data.drop([x_key], axis=1)
    for i in range(len(output[0])):
        data[f'{x_key}_{i}'] = cut(output, i)
    print(f'End model preprocess text key - {x_key}')
    return data


def get_first(string: str) -> str:
    result = string.split(' ')[0]
    if result == '':
        return string
    return result


def simple_text_preprocess(string: str) -> str:
    string = string.lower()
    return string.replace('ё', 'е')


def preprocess_area_name(data: DataFrame) -> DataFrame:
    key = "area_name"
    keyframe = data[key]
    keyframe = keyframe.apply(simple_text_preprocess)
    data[key] = keyframe
    return data


TOP_20_EMPLOYER = {
    "пятерочка",
    "перекресток",
    "билайн",
    "группа",
    "центр",
    "школа",
    "агентство",
    "компания",
    "гк",
    "сеть",
    "ип",
    "гбу",
    "кадровое",
    "jcat.ru",
    "hr",
    "университет",
    "skyeng",
    "лаборатория",
    "гбуз",
    "детский"
}


TOP_20_AREAS = {
    "москва",
    "санкт-,петербург"
    "новосибирск",
    "краснодар",
    "екатеринбург",
    "казань",
    "нижний новгород,"
    "ростов-,на-дону"
    "самара",
    "томск",
    "воронеж",
    "минск",
    "пермь",
    "челябинск",
    "уфа",
    "саратов",
    "тула",
    "калининград",
    "омск",
    "красноярск"
}


def is_top_20_employer(employer_name: str) -> int:
    if isinstance(employer_name, int):
        return employer_name
    if employer_name in TOP_20_EMPLOYER:
        return 1
    return 0


def is_top_20_area(area_name: str) -> int:
    if isinstance(area_name, int):
        return area_name
    if area_name in TOP_20_AREAS:
        return 1
    return 0


def preprocess_employer_name(data: DataFrame) -> DataFrame:
    key = "employer_name"
    keyframe = data[key]
    keyframe = keyframe.apply(simple_text_preprocess)
    keyframe = keyframe.apply(get_first)
    data[key] = keyframe
    return data


def preprocess_employer_name_to_int(data: DataFrame) -> DataFrame:
    key = "employer_name"
    keyframe = data[key]
    keyframe = keyframe.apply(is_top_20_employer)
    data[key] = keyframe
    return data


def preprocess_salary_gross(data: DataFrame) -> DataFrame:
    key = "salary_gross"
    data[key] = data[key].fillna(False)
    return data


def preprocess_area_name_to_int(data: DataFrame) -> DataFrame:
    key = "area_name"
    keyframe = data[key]
    keyframe = keyframe.apply(is_top_20_area)
    data[key] = keyframe
    return data


def add_prediction_by_name_desc(data: DataFrame) -> DataFrame:
    x = np.asarray(data[NAMES_AND_DESC_FEATURES]).astype('float32')
    model = load_name_desc_model()
    y = np.asarray(model.predict(x)).astype('float32')
    data[NAME_DESC_PREDICTION_KEY] = y
    return data


def add_nn_prediction_by_name_desc(data: DataFrame) -> DataFrame:
    x = np.asarray(data[NAMES_AND_DESC_FEATURES]).astype('float32')
    model = load_name_desc_nn_model(RU_NAME_DESC_MODELS_DIR)
    y = np.asarray(model.predict(x)).astype('float32')
    data[NAME_DESC_PREDICTION_KEY + "_nn"] = y
    return data


def add_prediction_by_categorical(data: DataFrame) -> DataFrame:
    original_data = data.copy()
    data = data.copy()
    data = data[CATEGORICAL_FEATURES]
    x = np.asarray(data).astype('bool')
    model = load_categorical_model()
    y = np.asarray(model.predict(x)).astype('float32')
    original_data[CATEGORICAL_KEY] = y
    return original_data


def add_nn_prediction_by_categorical(data: DataFrame) -> DataFrame:
    original_data = data.copy()
    data = data.copy()
    data = data[CATEGORICAL_FEATURES]
    x = np.asarray(data).astype('bool')
    model = load_categorical_nn_model(CATEGORICAL_DIR)
    y = np.asarray(model.predict(x)).astype('float32')
    original_data[CATEGORICAL_KEY + "_nn"] = y
    return original_data


def fill_salary_from(data: DataFrame) -> DataFrame:
    x_data = data.copy()
    x_data_index = x_data[SALARY_FROM_KEY].isna()
    x_data = x_data.drop(['created_at', 'published_at'], axis=1)
    x = x_data[x_data_index]

    x = x.drop([SALARY_FROM_KEY, 'id'], axis=1)
    x = np.asarray(x).astype('float32')
    x = create_data_to_eval_salary_from(x)
    x = np.asarray(x).astype('float32')
    model = load_salary_from_nn_model(EVAL_SALARY_FROM_RECOVER_MODELS_DIR)
    y = np.asarray(model.predict(x)).astype('float32')
    copy = data[x_data_index].copy()
    copy[SALARY_FROM_KEY] = y
    data[x_data_index] = copy
    return data


def is_over_or_2021(year: str) -> int:
    if int(year) >= 2021:
        return 1
    return 0


def to_one_got_month(month_number: str) -> list[int]:
    number: int = int(month_number)
    out = [0 for _ in range(12)]
    out[number - 1] = 1
    return out


def preprocess_months(data: DataFrame, key: str) -> DataFrame:
    months = np.asarray(data[key]).astype('str')
    one_hot = list(map(lambda x: to_one_got_month(x), months))
    for i in range(len(one_hot[0])):
        data[f'{key}_{i}'] = cut(one_hot, i)
    data = data.drop([key], axis=1)
    return data


def preprocess_date(data: DataFrame, key: str) -> DataFrame:
    keyframe = data[key]
    date_array = np.asarray(keyframe.apply(get_first)).astype('str')
    date_array = list(map(lambda x: x.split('-'), date_array))
    data[f'{key}_year'] = cut(date_array, 0)
    data[f'{key}_month'] = cut(date_array, 1)
    data = preprocess_months(data, f'{key}_month')
    data[f'{key}_year'] = data[f'{key}_year'].apply(is_over_or_2021)
    data = data.drop([key], axis=1)
    return data


def preprocess_with_models(data: DataFrame) -> DataFrame:
    original_data = data
    data = data.copy()
    data = data.drop(['id'], axis=1)
    data = np.asarray(data).astype('float32')
    nn_eval_model = load_eval_nn_model(EVAL_MODELS_DIR)
    eval_nn_result = nn_eval_model.predict(data)

    eval_model = load_eval_model()
    eval_result = eval_model.predict(data)

    result = DataFrame()
    result['id'] = original_data['id']
    result['eval'] = eval_result
    result['nn_eval'] = eval_nn_result
    return result


def preprocess_data(data: DataFrame,
                    skip_drop: bool = False,
                    skip_text_preprocessing: bool = False,
                    skip_models_text_preprocessing: bool = False,
                    skip_name_desc_prediction: bool = False,
                    skip_simple_mappings: bool = False,
                    skip_filling: bool = False,
                    skip_date_preprocess: bool = False,
                    skip_categorical_predictions: bool = False,
                    skip_model_preprocess: bool = False) -> DataFrame:
    data = data.copy()
    if not skip_drop:
        try:
            data = data.drop(['area_id', 'salary_currency'], axis=1)
        except Exception:
            pass
    if not skip_text_preprocessing:  # 1
        data = preprocess_text(data, 'name')
        data = preprocess_text(data, 'description')

        description = data['description']
        lengths = description.apply(lambda x: math.log(len(x)))
        data['desc_length'] = lengths

        data = preprocess_area_name(data)
        data = preprocess_employer_name(data)
    if not skip_models_text_preprocessing:  # 2
        data = preprocess_text_with_rubert(data, RU_BERT_BASED_NAME_CHECKPOINT_DIR, 'name')
        data = preprocess_text_with_rubert(data, RU_BERT_BASED_DESCRIPTION_CHECKPOINT_DIR, 'description')
    if not skip_name_desc_prediction:  # 3
        data = add_prediction_by_name_desc(data)
        data = add_nn_prediction_by_name_desc(data)
    if not skip_simple_mappings:  # 4
        data = preprocess_salary_gross(data)
        data = preprocess_employer_name_to_int(data)
        data = preprocess_area_name_to_int(data)
        data.salary_gross = data.salary_gross.replace({True: 1, False: 0})
        data.has_test = data.has_test.replace({True: 1, False: 0})
        data.response_letter_required = data.response_letter_required.replace({True: 1, False: 0})
    data.to_csv('save.csv')
    if not skip_filling:  # 5
        data = fill_salary_from(data)
    if not skip_date_preprocess:  # 6
        data = preprocess_date(data, 'published_at')
        data = preprocess_date(data, 'created_at')
    if not skip_categorical_predictions:  # 7
        data = add_prediction_by_categorical(data)
        data = add_nn_prediction_by_categorical(data)
    if not skip_model_preprocess:  # 8
        data = preprocess_with_models(data)
    return data


def inverse(y_to_inverse: Iterable) -> Iterable:
    y_to_inverse = np.asarray(list(map(lambda x: math.exp(x), y_to_inverse))).astype('float32')
    return y_to_inverse
