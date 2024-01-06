import os

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from support.constants import (TARGET_NAME, RU_BERT_BASED_NAME_CHECKPOINT_DIR, RU_BERT_BASED_DESCRIPTION_CHECKPOINT_DIR,
                               NAMES_PRE_FEATURES, DESC_PRE_FEATURES, BERT_MODEL_OUT_SIZE)
from support.functions import load_x_prepared_train_data, load_y_train_norm_data, smape_loss, split_to_batches

import tensorflow as tf
from tensorflow.python.keras import activations


class RuBertBasedModel(tf.keras.Model):
    def __init__(self, is_work: bool = False):
        super(RuBertBasedModel, self).__init__(name='')
        self.is_work = is_work
        self.first = tf.keras.layers.Dense(32, activation=activations.linear)
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.second = tf.keras.layers.Dense(BERT_MODEL_OUT_SIZE, activation=activations.softplus)
        self.forth = tf.keras.layers.Dense(1, activation=activations.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        x = self.first(inputs)
        x = self.dropout(x)
        x = self.second(x)
        if not self.is_work:
            x = self.forth(x)
        return x


def create_and_learn_rubert_text_models(x, y,
                                        model_name: str,
                                        checkpoints_dir: str) -> tf.keras.Model:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = RuBertBasedModel()
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


def load_work_rubert_text_model(checkpoint_dir: str) -> RuBertBasedModel:
    loaded_model = RuBertBasedModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def main():
    x_data = load_x_prepared_train_data()
    y_data = load_y_train_norm_data()

    y_data = np.asarray(y_data[TARGET_NAME]).astype('float32')
    create_and_learn_rubert_text_models(np.asarray(x_data[NAMES_PRE_FEATURES]).astype('float32'), y_data,
                                        "ru_name_model", RU_BERT_BASED_NAME_CHECKPOINT_DIR)
    create_and_learn_rubert_text_models(np.asarray(x_data[DESC_PRE_FEATURES]).astype('float32'), y_data,
                                        "ru_desc_model", RU_BERT_BASED_DESCRIPTION_CHECKPOINT_DIR)


def model_work(model, input_texts, batch_size) -> list[np.array]:
    output = []
    batches = split_to_batches(input_texts, batch_size)
    i = 0
    for batch in batches:
        output.extend(model(batch).numpy())
        i += 1
        print(f'Model preprocess {i}/{len(batches)} batches')
    return output


def cut(arrays, idx) -> list:
    result = []
    for array in arrays:
        result.append(array[idx])
    return result


def preprocess_text_with_rubert(data: DataFrame, checkpoint_dir: str, inputs, x_key: str) -> DataFrame:
    rubert_encode_model = load_work_rubert_text_model(checkpoint_dir)
    batch_size = 128
    output = model_work(rubert_encode_model, inputs, batch_size)
    for i in range(len(output[0])):
        data[f'{x_key}_{i}'] = cut(output, i)
    print(f'End model preprocess text key - {x_key}')
    return data


if __name__ == '__main__':
    main()

    # x_data = load_x_prepared_train_data()
    # x_data = preprocess_text_with_rubert(x_data, RU_BERT_BASED_NAME_CHECKPOINT_DIR,
    #                                      np.asarray(x_data[NAMES_PRE_FEATURES]).astype('float64'), 'name')
    # x_data = x_data.drop(NAMES_PRE_FEATURES[BERT_MODEL_OUT_SIZE:], axis=1)
    # x_data = preprocess_text_with_rubert(x_data, RU_BERT_BASED_DESCRIPTION_CHECKPOINT_DIR,
    #                                      np.asarray(x_data[DESC_PRE_FEATURES]).astype('float64'), 'description')
    # x_data = x_data.drop(DESC_PRE_FEATURES[BERT_MODEL_OUT_SIZE:], axis=1)
    # x_data.to_csv('buf.csv', index=False)
