import os

import numpy as np
import tensorflow as tf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import activations

from models_creation import load_salary_from_model
from support.constants import SALARY_FROM_KEY, SALARY_FROM_RECOVER_MODELS_DIR, EVAL_SALARY_FROM_RECOVER_MODELS_DIR
from support.functions import load_x_prepared_train_data, smape_loss


class SalaryFromModel(tf.keras.Model):
    def __init__(self, is_work: bool = False):
        super(SalaryFromModel, self).__init__(name='')
        self.is_work = is_work
        self.first = tf.keras.layers.Dense(64, activation=activations.linear)
        self.second = tf.keras.layers.Dense(32, activation=activations.softplus)
        self.forth = tf.keras.layers.Dense(1, activation=activations.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        x = self.first(inputs)
        x = self.second(x)
        x = self.forth(x)
        return x


def create_and_learn_name_desc_models(x, y,
                                      model_name: str,
                                      checkpoints_dir: str) -> tf.keras.Model:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = SalaryFromModel()
    checkpoint_name = model_name + '-{epoch:04d}.ckpt'
    checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_name)
    model.compile(tf.keras.optimizers.Adam(), loss=smape_loss)
    model.save_weights(checkpoint_filepath.format(epoch=0))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    model.fit(x_train, y_train, verbose=1, validation_data=(x_test, y_test),
              batch_size=128, epochs=500, shuffle=True, callbacks=[cp_callback])
    return model


def load_salary_from_nn_model(checkpoint_dir: str) -> SalaryFromModel:
    loaded_model = SalaryFromModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def main():
    x_data = load_x_prepared_train_data()
    x_data = x_data[x_data[SALARY_FROM_KEY].notna()]
    y_data = x_data[SALARY_FROM_KEY]

    x_data = x_data.drop([SALARY_FROM_KEY, 'id', 'created_at', 'published_at'], axis=1)
    x_data = np.asarray(x_data).astype('float32')
    y_data = np.asarray(y_data).astype('float32')
    create_and_learn_name_desc_models(x_data, y_data, "salary_from_recover", SALARY_FROM_RECOVER_MODELS_DIR)


def create_data_to_eval_salary_from(x_data):
    result = load_salary_from_model().predict(x_data)
    nn_result = load_salary_from_nn_model(SALARY_FROM_RECOVER_MODELS_DIR).predict(x_data)
    res = DataFrame()
    res['sf'] = result
    res['nn_sf'] = nn_result
    return res


def main_eval():
    x_data = load_x_prepared_train_data()
    x_data = x_data[x_data[SALARY_FROM_KEY].notna()]
    y_data = x_data[SALARY_FROM_KEY]

    x_data = x_data.drop([SALARY_FROM_KEY, 'id', 'created_at', 'published_at'], axis=1)
    x_data = np.asarray(x_data).astype('float32')
    x_data = create_data_to_eval_salary_from(x_data)

    x_data = np.asarray(x_data).astype('float32')
    y_data = np.asarray(y_data).astype('float32')
    create_and_learn_name_desc_models(x_data, y_data, "eval_salary_from_recover", EVAL_SALARY_FROM_RECOVER_MODELS_DIR)


if __name__ == '__main__':
    main_eval()
