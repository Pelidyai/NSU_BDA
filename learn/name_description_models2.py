import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import activations

from support.constants import TARGET_NAME, NAMES_AND_DESC_FEATURES, RU_NAME_DESC_MODELS_DIR
from support.functions import load_x_prepared_train_data, load_y_train_norm_data


class NameDescModel(tf.keras.Sequential):
    def __init__(self, is_work: bool = False):
        super(NameDescModel, self).__init__(name='')
        self.is_work = is_work
        self.add(tf.keras.layers.Dense(128, activation=activations.linear))
        self.add(tf.keras.layers.Dropout(rate=0.1))

        self.add(tf.keras.layers.Dense(32, activation=activations.softplus))
        self.add(tf.keras.layers.Dropout(rate=0.05))

        self.add(tf.keras.layers.Dense(1, activation=activations.leaky_relu))


def create_and_learn_name_desc_models(x, y,
                                      model_name: str,
                                      checkpoints_dir: str) -> tf.keras.Model:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = NameDescModel()
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
              batch_size=32, epochs=5000, shuffle=True, callbacks=[cp_callback])
    return model


def main():
    x_data = load_x_prepared_train_data()
    y_data = load_y_train_norm_data()

    x_data = np.asarray(x_data[NAMES_AND_DESC_FEATURES]).astype('float32')
    y_data = np.asarray(y_data[TARGET_NAME]).astype('float32')
    create_and_learn_name_desc_models(x_data, y_data,
                                      "name_desc_norm", RU_NAME_DESC_MODELS_DIR)


if __name__ == '__main__':
    main()
