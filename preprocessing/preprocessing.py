from pandas import DataFrame

from support.functions import prepare_text


def preprocess_text(data: DataFrame, key: str) -> DataFrame:
    key_frame = data[key]
    key_frame = key_frame.fillna('empty')
    print(f'Start preprocess key - {key}')
    prepared_frame = key_frame.apply(prepare_text)
    print(f'End preprocess key - {key}')
    data[key] = prepared_frame
    return data


def preprocess_data(data: DataFrame) -> DataFrame:
    data = preprocess_text(data, 'name')
    data = preprocess_text(data, 'description')
    return data
