import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import activations

from support.constants import TARGET_NAME, FINAL_MODELS_DIR
from support.functions import load_x_prepared_train_data, smape_loss, load_y_train_norm_data


class FinalModel(tf.keras.Model):
    def __init__(self, is_work: bool = False):
        super(FinalModel, self).__init__(name='')
        self.is_work = is_work
        self.first = tf.keras.layers.Dense(64, activation=activations.linear)
        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        self.second = tf.keras.layers.Dense(32, activation=activations.softplus)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.05)
        self.forth = tf.keras.layers.Dense(1, activation=activations.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        x = self.first(inputs)
        x = self.dropout(x)
        x = self.second(x)
        x = self.dropout2(x)
        x = self.forth(x)
        return x


def create_and_learn_evaluate_models(x, y,
                                     model_name: str,
                                     checkpoints_dir: str) -> tf.keras.Model:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = FinalModel()
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
              batch_size=128, epochs=5000, shuffle=True, callbacks=[cp_callback])
    return model


def main():
    x_data = load_x_prepared_train_data()
    y_data = load_y_train_norm_data()[TARGET_NAME]
    x_data = x_data.drop(['id'], axis=1)

    x_data = np.asarray(x_data).astype('float32')
    y_data = np.asarray(y_data).astype('float32')
    create_and_learn_evaluate_models(x_data, y_data, "final_model", FINAL_MODELS_DIR)


if __name__ == '__main__':
    main()
