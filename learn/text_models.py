import os.path

import numpy as np
from pandas import DataFrame
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models_creation import BertBasedModel
from support.constants import TARGET_NAME, BERT_BASED_NAME_CHECKPOINT_DIR, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR
from support.functions import load_x_prepared_train_data, load_y_train_norm_data, smape_loss, load_y_train_data


def create_and_learn_text_model(data: DataFrame,
                                x_key: str,
                                model_name: str,
                                checkpoints_dir: str) -> tf.keras.Model:
    x = data[x_key]
    y = data[TARGET_NAME]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = BertBasedModel()
    checkpoint_name = model_name + '-{epoch:04d}.ckpt'
    checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_name)
    model.compile(tf.keras.optimizers.Adam(), loss=smape_loss)
    model.save_weights(checkpoint_filepath.format(epoch=0))
    x_train = np.asarray(x_train).astype('str')
    x_test = np.asarray(x_test).astype('str')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    model.fit(x_train, y_train, verbose=1, validation_data=(x_test, y_test),
              batch_size=128, epochs=10, shuffle=True, callbacks=[cp_callback])
    return model


def main():
    data = load_x_prepared_train_data()
    data[TARGET_NAME] = load_y_train_norm_data()[TARGET_NAME]
    create_and_learn_text_model(data, 'name', "bert_name_to_8_1_norm", BERT_BASED_NAME_CHECKPOINT_DIR)
    create_and_learn_text_model(data, 'description', "bert_desc_to_8_1_norm", BERT_BASED_DESCRIPTION_CHECKPOINT_DIR)


if __name__ == '__main__':
    main()
