import os
from pathlib import Path

splitted: list[str] = os.getcwd().split(os.sep)[:-1]
splitted[0] += os.sep
file_dir: str = os.path.join(*splitted)


def get_path(dir_to_search: str, file_to_search: str):
    try:
        return str(next(Path(dir_to_search).rglob(file_to_search)))
    except StopIteration:
        return os.path.join(dir_to_search, file_to_search)


SALARY_FROM_RECOVER_MODELS_DIR: str = get_path(file_dir, 'models/salary_from')
NAME_DESC_MODELS_DIR: str = get_path(file_dir, 'models/name_desc')
RU_NAME_DESC_MODELS_DIR: str = get_path(file_dir, 'models/ru_name_desc')
CATEGORICAL_DIR: str = get_path(file_dir, 'models/categorical')
EVAL_MODELS_DIR: str = get_path(file_dir, 'models/eval')
FINAL_MODELS_DIR: str = get_path(file_dir, 'models/final')
ENSEMBLE_MODELS_DIR: str = get_path(file_dir, 'models/ensemble')
FINAL_MODELS_CHECKPOINT_DIR: str = get_path(file_dir, 'models/final_checkpoints')

BERT_BASED_NAME_CHECKPOINT_DIR: str = get_path(file_dir, 'models/name')
BERT_BASED_DESCRIPTION_CHECKPOINT_DIR: str = get_path(file_dir, 'models/description')

RU_BERT_DIR: str = get_path(file_dir, 'rubert-tiny')

RU_BERT_BASED_NAME_CHECKPOINT_DIR: str = get_path(file_dir, 'models/ru_name')
RU_BERT_BASED_DESCRIPTION_CHECKPOINT_DIR: str = get_path(file_dir, 'models/ru_description')
RU_BERT_BASED_NAME_DESCRIPTION_CHECKPOINT_DIR: str = get_path(file_dir, 'models/ru_name_desc')


MLP_MODEL_PATH: str = get_path(file_dir, 'models/32_32_8mlp.pickaim')
GRAD_MODEL_PATH: str = get_path(file_dir, 'models/grad.pickaim')
RFR_MODEL_PATH: str = get_path(file_dir, 'models/simple-forest.pickaim')

SCALER_PATH: str = get_path(file_dir, 'models/scaler/scaler.pickle')

X_TRAIN_PATH: str = get_path(file_dir, "data/X_train.csv")
PREP_X_TRAIN_PATH: str = get_path(file_dir, "data/prep_X_train.csv")
X_TEST_PATH: str = get_path(file_dir, "data/X_test.csv")
PREP_X_TEST_PATH: str = get_path(file_dir, "data/prep_X_test.csv")

Y_TRAIN_PATH: str = get_path(file_dir, "data/Y_train.csv")
Y_TRAIN_NORM_PATH: str = get_path(file_dir, "data/Y_train_norm.csv")

BERT_MODEL_OUT_SIZE = 8

NAMES_AND_DESC_FEATURES = [*[f"name_{i}" for i in range(BERT_MODEL_OUT_SIZE)],
                           *[f"description_{i}" for i in range(BERT_MODEL_OUT_SIZE)]]

NAMES_PRE_FEATURES = [f"name_{i}" for i in range(312)]

DESC_PRE_FEATURES = [f"description_{i}" for i in range(312)]

TRAIN_FEATURE = ['name', 'has_test', 'response_letter_required', 'salary_from', 'salary_currency', 'salary_gross',
                 'published_at', 'created_at', 'employer_name', 'description', 'area_id', 'area_name']

CATEGORICAL_FEATURES = ['has_test', 'response_letter_required', 'salary_gross', 'employer_name',
                        'area_name', 'published_at_year', 'created_at_year',
                        *[f"published_at_month_{i}" for i in range(12)],
                        *[f"created_at_month_{i}" for i in range(12)]]
TARGET_NAME = 'salary_to'

NAME_DESC_PREDICTION_KEY = 'salary_to_by_name_desc'
CATEGORICAL_KEY = 'salary_to_by_categorical'
SALARY_FROM_KEY = 'salary_from'
