import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras import activations

import tensorflow_text as text

from support.bert import BERT_PREPROCESS_LINK, BERT_ENCODER_LINK

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


#
# class BertLayer(tf.keras.layers.Layer):
#     def __init__(self, num_outputs):
#       super(BertLayer, self).__init__()
#       self.num_outputs = num_outputs
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
#
#     def call(self, inputs, *args, **kwargs):
#         preprocessing_layer = hub.KerasLayer(BERT_PREPROCESS_LINK, name='preprocessing')
#         encoder_inputs = preprocessing_layer(inputs)
#         encoder = hub.KerasLayer(BERT_ENCODER_LINK, trainable=True, name='BERT_encoder')
#         outputs = encoder(encoder_inputs)
#         return outputs['pooled_output']
#
#
# def build_base_text_processing_model() -> tf.keras.Model:
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#     net = BertLayer(768)(text_input)
#     net = tf.keras.layers.Dropout(0.2)(net)
#     net = tf.keras.layers.Dense(5, activation=activations.sigmoid)(net)
#     net = tf.keras.layers.Dense(1, activation=activations.relu)(net)
#     return tf.keras.Model(text_input, net)


class BertBasedModel(tf.keras.Model):
    def __init__(self, is_work: bool = False):
        super(BertBasedModel, self).__init__(name='')
        self.is_work = is_work
        self.bert = _build_bert()
        self.first = tf.keras.layers.Dense(64, activation=activations.linear)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.second = tf.keras.layers.Dense(16, activation=activations.leaky_relu)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.third = tf.keras.layers.Dense(8, activation=activations.leaky_relu)
        self.forth = tf.keras.layers.Dense(1, activation=activations.softplus)

    def call(self, inputs, training=None, mask=None):
        x = self.bert(inputs)
        x = self.first(x)
        x = self.dropout1(x)
        x = self.second(x)
        x = self.dropout2(x)
        x = self.third(x)
        if not self.is_work:
            x = self.forth(x)
        return x


def load_work_text_model(checkpoint_dir: str) -> BertBasedModel:
    loaded_model = BertBasedModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


# model = load_work_text_model(BERT_BASED_NAME_CHECKPOINT_DIR)
# print(model(tf.constant(['hello'])))

# model = BertBasedModel()
# print(model(tf.constant(['hello'])))
