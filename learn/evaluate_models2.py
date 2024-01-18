import os

import numpy as np
import tensorflow as tf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import activations

from support.constants import TARGET_NAME, EVAL_MODELS_DIR
from support.functions import load_x_prepared_train_data, smape_loss, load_y_train_data, load_y_train_norm_data, \
    scheduler


class EvaluateModel(tf.keras.models.Sequential):
    def __init__(self, is_work: bool = False):
        super(EvaluateModel, self).__init__(name='')
        self.add(tf.keras.layers.Dense(128, activation=activations.linear))
        self.add(tf.keras.layers.Dropout(rate=0.1))

        self.add(tf.keras.layers.Dense(32, activation=activations.leaky_relu))
        self.add(tf.keras.layers.Dropout(rate=0.1))

        self.add(tf.keras.layers.Dense(1, activation=activations.softplus))


def load_nn():
    loaded_model = EvaluateModel(is_work=True)
    latest = tf.train.latest_checkpoint(EVAL_MODELS_DIR)
    loaded_model.load_weights(latest)
    return loaded_model


def create_and_learn_evaluate_models(x, y,
                                     model_name: str,
                                     checkpoints_dir: str) -> tf.keras.Model:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = load_nn()
    checkpoint_name = model_name + '-{epoch:04d}.ckpt'
    checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_name)
    model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.MAPE)
    model.save_weights(checkpoint_filepath.format(epoch=0))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    model.fit(x_train, y_train, verbose=1, validation_data=(x_test, y_test),
              batch_size=16, epochs=5000, shuffle=True, callbacks=[cp_callback,
                                                                   tf.keras.callbacks.LearningRateScheduler(scheduler)])
    #
    return model


def main():
    x_data = load_x_prepared_train_data()
    y_data = load_y_train_norm_data()[TARGET_NAME]
    x_data = x_data.drop(['id'], axis=1)

    x_data = np.asarray(x_data).astype('float32')
    y_data = np.asarray(y_data).astype('float32')
    create_and_learn_evaluate_models(x_data, y_data, "eval_model", EVAL_MODELS_DIR)


if __name__ == '__main__':
    main()
    # x_data = load_x_prepared_train_data()
    # y_data = load_y_train_norm_data()[TARGET_NAME]
    # x_data = x_data.drop(['id'], axis=1)
    # correlations = x_data.corrwith(y_data)
    # correlations = correlations.apply(lambda x: abs(x))
    # frame = DataFrame()
    # frame['key'] = correlations.index
    # frame['values'] = correlations.values
    # frame = frame[frame['values'] > 0.2]
    # result = np.asarray(frame['key']).astype('str')
    # print("', '".join(result))
