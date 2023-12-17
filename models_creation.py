import pickle
from typing import Any

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.neural_network import MLPRegressor
from tensorflow.python.keras import activations

import tensorflow_text as text

from support.bert import BERT_PREPROCESS_LINK, BERT_ENCODER_LINK
from support.constants import NAME_DESC_MODELS_DIR

tf.get_logger().setLevel('ERROR')


def _build_bert() -> tf.keras.Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(BERT_PREPROCESS_LINK, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(BERT_ENCODER_LINK, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    result_model = tf.keras.Model(text_input, net)
    result_model.trainable = False
    return result_model


BERT_MODEL_OUT_SIZE = 32


class BertBasedModel(tf.keras.Model):
    def __init__(self, is_work: bool = False):
        super(BertBasedModel, self).__init__(name='')
        self.is_work = is_work
        self.bert = _build_bert()
        self.first = tf.keras.layers.Dense(BERT_MODEL_OUT_SIZE, activation=activations.softplus)
        self.forth = tf.keras.layers.Dense(1, activation=activations.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        x = self.bert(inputs)
        x = self.first(x)
        if not self.is_work:
            x = self.forth(x)
        return x


def load_work_text_model(checkpoint_dir: str) -> BertBasedModel:
    loaded_model = BertBasedModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def create_name_description_model() -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(8, 8, 1))


def load_name_desc_model() -> Any:
    with open(NAME_DESC_MODELS_DIR + "/best.pickaim", 'rb') as file:
        return pickle.load(file)
