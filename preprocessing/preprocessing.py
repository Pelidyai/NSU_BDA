import math
from typing import Iterable

import numpy as np
from pandas import DataFrame

from models_creation import load_work_text_model, load_name_desc_model
from support.constants import BERT_BASED_NAME_CHECKPOINT_DIR, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR, \
    NAME_DESC_PREDICTION_KEY, NAMES_AND_DESC_FEATURES
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
    data = data.drop(NAMES_AND_DESC_FEATURES, axis=1)
    data[NAME_DESC_PREDICTION_KEY] = y
    return data


def preprocess_data(data: DataFrame,
                    skip_drop: bool = False,
                    skip_text_preprocessing: bool = False,
                    skip_models_text_preprocessing: bool = False,
                    skip_name_desc_prediction: bool = False,
                    skip_simple_mappings: bool = False) -> DataFrame:
    data = data.copy()
    if not skip_drop:
        try:
            data = data.drop(['area_id', 'published_at', 'created_at','salary_currency'], axis=1)
        except Exception:
            pass
    if not skip_text_preprocessing:  # 1
        data = preprocess_text(data, 'name')
        data = preprocess_text(data, 'description')
        data = preprocess_area_name(data)
        data = preprocess_employer_name(data)
    if not skip_models_text_preprocessing:  # 2
        data = preprocess_text_with_model(data, BERT_BASED_NAME_CHECKPOINT_DIR, 'name')
        data = preprocess_text_with_model(data, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR, 'description')
    if not skip_name_desc_prediction:  # 3
        data = add_prediction_by_name_desc(data)
    if not skip_simple_mappings:  # 4
        data = preprocess_salary_gross(data)
        data = preprocess_employer_name_to_int(data)
        data = preprocess_area_name_to_int(data)
        data.salary_gross = data.salary_gross.replace({True: 1, False: 0})
        data.has_test = data.has_test.replace({True: 1, False: 0})
        data.response_letter_required = data.response_letter_required.replace({True: 1, False: 0})
    return data


def logo_normalize(data: DataFrame, target: str) -> DataFrame:
    normalized = data.copy()
    normalized[target] = normalized[target].apply(math.log)
    return normalized


def inverse(y_to_inverse: Iterable) -> Iterable:
    y_to_inverse = np.asarray(list(map(lambda x: math.exp(x), y_to_inverse))).astype('float32')
    return y_to_inverse
