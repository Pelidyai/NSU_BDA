import math
from typing import Iterable

import numpy as np
from pandas import DataFrame

from models_creation import load_work_text_model
from support.constants import BERT_BASED_NAME_CHECKPOINT_DIR, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR
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
    return string.split(' ')[0]


def preprocess_employer_name(data: DataFrame) -> DataFrame:
    key = "employer_name"
    data = preprocess_text(data, key)
    keyframe = data[key]
    keyframe = keyframe.apply(get_first)
    data[key] = keyframe
    return data

    # count_series = data[key].value_counts()
    # counts = count_series.to_numpy()
    # max_idx = max(0, len(counts) - 1)
    # top_3_frontier = counts(min(3, max_idx))
    # top_100_frontier = counts(min(100, max_idx))
    # top_3_flags = []
    # top_100_flags = []
    # for employer_name in data[key]:
    #     top_3_flags.append(count_series[employer_name].to_numpy() > top_3_frontier)
    #     top_100_flags.append(count_series[employer_name].to_numpy() > top_100_frontier)
    # counts = None


def preprocess_data(data: DataFrame,
                    skip_drop: bool = False,
                    skip_text_preprocessing: bool = False,
                    skip_models_text_preprocessing: bool = False,
                    skip_obj_to_cat: bool = False) -> DataFrame:
    data = data.copy()
    if not skip_drop:
        try:
            data = data.drop(['area_id', 'published_at', 'created_at', 'salary_currency'], axis=1)
        except Exception:
            pass
    if not skip_text_preprocessing:
        data = preprocess_text(data, 'name')
        data = preprocess_text(data, 'description')
        data = preprocess_text(data, 'area_name')
        data = preprocess_employer_name(data)
    if not skip_models_text_preprocessing:
        data = preprocess_text_with_model(data, BERT_BASED_NAME_CHECKPOINT_DIR, 'name')
        data = preprocess_text_with_model(data, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR, 'description')
    # if not skip_obj_to_cat:
    #     data = preprocess_employer_name(data)
    return data


def logo_normalize(data: DataFrame, target: str) -> DataFrame:
    normalized = data.copy()
    normalized[target] = normalized[target].apply(math.log)
    return normalized


def inverse(y_to_inverse: Iterable) -> Iterable:
    y_to_inverse = np.asarray(list(map(lambda x: math.exp(x), y_to_inverse))).astype('float32')
    return y_to_inverse
