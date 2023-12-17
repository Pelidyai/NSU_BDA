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


def preprocess_text_with_model(data: DataFrame, checkpoint_dir: str, x_key: str) -> DataFrame:
    loaded_model = load_work_text_model(checkpoint_dir)
    input_texts = data[x_key].to_numpy()
    batch_size = 128
    output = []
    batches = split_to_batches(input_texts, batch_size)
    i = 0
    print(f'Start model preprocess text key - {x_key}')
    for batch in batches:
        output.extend(loaded_model(batch).numpy())
        i += 1
        print(f'Model preprocess {i}/{len(batches)} batches')
    data = data.drop([x_key], axis=1)
    for i in range(len(output[0])):
        data[f'{x_key}_{i}'] = cut(output, i)
    print(f'End model preprocess text key - {x_key}')
    return data


def preprocess_data(data: DataFrame,
                    skip_drop: bool = False,
                    skip_text_preprocessing: bool = False,
                    skip_models_text_preprocessing: bool = False) -> DataFrame:
    data = data[:32]
    if not skip_drop:
        try:
            data = data.drop(['area_id', 'published_at', 'created_at', 'salary_currency'], axis=1)
        except Exception:
            pass
    if not skip_text_preprocessing:
        data = preprocess_text(data, 'name')
        data = preprocess_text(data, 'description')
        data = preprocess_text(data, 'area_name')
        data = preprocess_text(data, 'employer_name')
    if not skip_models_text_preprocessing:
        data = preprocess_text_with_model(data, BERT_BASED_NAME_CHECKPOINT_DIR, 'name')
        data = preprocess_text_with_model(data, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR, 'description')
    return data
