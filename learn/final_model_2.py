import os.path

import numpy as np
from keras import activations
from pandas import DataFrame
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing.preprocessing import model_work
from support.constants import TARGET_NAME, FINAL_MODELS_CHECKPOINT_DIR, SALARY_FROM_KEY, NAME_DESC_PREDICTION_KEY
from support.functions import load_x_prepared_train_data, load_y_train_norm_data, smape_loss, load_y_train_data


class FinalModel(tf.keras.Model):
    def __init__(self,):
        super(FinalModel, self).__init__(name='')
        self.first = tf.keras.layers.Dense(21, activation=activations.relu)
        self.second = tf.keras.layers.Dense(49, activation=activations.exponential)
        self.third = tf.keras.layers.Dense(49, activation=activations.softplus)
        self.forth = tf.keras.layers.Dense(1, activation=activations.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        x = self.first(inputs)
        x = self.second(x)
        x = self.third(x)
        x = self.forth(x)
        return x


def load_model_to_tune() -> FinalModel:
    model = FinalModel()
    latest = tf.train.latest_checkpoint('../models/save/final_checkpoints')  # 0.0118
    model.load_weights(latest)
    return model


def create_and_learn_text_model(x_data: DataFrame,
                                y_data: DataFrame,
                                model_name: str,
                                checkpoints_dir: str) -> tf.keras.Model:
    x = np.asarray(x_data).astype('float32')
    y = np.asarray(y_data).astype('float32')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = load_model_to_tune()
    checkpoint_name = model_name + '-{epoch:04d}.ckpt'
    checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_name)
    model.compile(tf.keras.optimizers.Adam(), loss=smape_loss)
    model.save_weights(checkpoint_filepath.format(epoch=0))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_weights_only=True)
    model.fit(x_train, y_train, verbose=1, validation_data=(x_test, y_test),
              batch_size=128, epochs=10_000, shuffle=True, callbacks=[cp_callback])
    return model


def main():
    x_data = load_x_prepared_train_data()
    x_data = x_data.drop(['id'], axis=1)
    # x_data = x_data[[SALARY_FROM_KEY, NAME_DESC_PREDICTION_KEY]]
    y_data = load_y_train_norm_data()[TARGET_NAME]
    create_and_learn_text_model(x_data, y_data, "final", FINAL_MODELS_CHECKPOINT_DIR)


def main_test():
    x_data = load_x_prepared_train_data()
    x_data = x_data.drop(['id'], axis=1)
    # x_data = x_data[[SALARY_FROM_KEY, NAME_DESC_PREDICTION_KEY]]
    y_data = load_y_train_norm_data()[TARGET_NAME]
    x = np.asarray(x_data).astype('float32')
    y = np.asarray(y_data).astype('float32')
    model = FinalModel()
    latest = tf.train.latest_checkpoint('../models/save/final_checkpoints/')
    model.load_weights(latest)
    pred = model_work(model, x, 128)
    pred = np.reshape(pred, (1, -1))[0]
    print(np.asarray(smape_loss(y, pred)).astype('float32').mean())


if __name__ == '__main__':
    main()
