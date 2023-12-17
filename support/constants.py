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


SALARY_GROSS_RECOVER_MODELS_DIR: str = get_path(file_dir, 'models/salary_gross')
NAME_DESC_MODELS_DIR: str = get_path(file_dir, 'models/name_desc')

BERT_BASED_NAME_CHECKPOINT_DIR: str = get_path(file_dir, 'models/name')
BERT_BASED_DESCRIPTION_CHECKPOINT_DIR: str = get_path(file_dir, 'models/description')

SCALER_PATH: str = get_path(file_dir, 'models/scaler/scaler.pickle')


X_TRAIN_PATH: str = get_path(file_dir, "data/X_train.csv")
PREP_X_TRAIN_PATH: str = get_path(file_dir, "data/prep_X_train.csv")
X_TEST_PATH: str = get_path(file_dir, "data/X_test.csv")

Y_TRAIN_PATH: str = get_path(file_dir, "data/Y_train.csv")
Y_TRAIN_NORM_PATH: str = get_path(file_dir, "data/Y_train_norm.csv")

TRAIN_FEATURE = ['name', 'has_test', 'response_letter_required', 'salary_from', 'salary_currency', 'salary_gross',
                 'published_at', 'created_at', 'employer_name', 'description', 'area_id', 'area_name']
TARGET_NAME = 'salary_to'

