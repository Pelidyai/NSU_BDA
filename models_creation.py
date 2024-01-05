import pickle
from typing import Any

import tensorflow as tf
import tensorflow_hub as hub
import torch
from sklearn.neural_network import MLPRegressor
from tensorflow.python.keras import activations
from transformers import AutoTokenizer, AutoModel

from learn.categorical_models2 import CategoricalModel
from learn.evaluate_model2 import EvaluateModel
from learn.final_model_2 import FinalModel
from learn.name_description_models2 import NameDescModel
from support.bert import BERT_PREPROCESS_LINK, BERT_ENCODER_LINK
from support.constants import NAME_DESC_MODELS_DIR, BERT_MODEL_OUT_SIZE, SALARY_FROM_RECOVER_MODELS_DIR, \
    CATEGORICAL_DIR, FINAL_MODELS_DIR, ENSEMBLE_MODELS_DIR, RU_BERT_DIR, EVAL_MODELS_DIR

tf.get_logger().setLevel('ERROR')


def _build_bert() -> tf.keras.Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(BERT_PREPROCESS_LINK, name='preprocessing', trainable=False)
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(BERT_ENCODER_LINK, trainable=False, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    result_model = tf.keras.Model(text_input, net)
    result_model.trainable = False
    return result_model


def _build_rubert():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    # text_input = np.asarray(text_input).astype('str')
    tokenizer = AutoTokenizer.from_pretrained(RU_BERT_DIR)
    t = tokenizer(text_input, padding=True, truncation=True, return_tensors='pt')
    ru_bert = AutoModel.from_pretrained(RU_BERT_DIR)
    with torch.no_grad():
        model_output = ru_bert(**{k: v.to(ru_bert.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    net = torch.nn.functional.normalize(embeddings)
    result_model = tf.keras.Model(text_input, net)
    result_model.trainable = False
    return result_model


class BertBasedModel(tf.keras.Model):
    def __init__(self, is_work: bool = False):
        super(BertBasedModel, self).__init__(name='')
        self.is_work = is_work
        self.bert = _build_bert()
        self.bert.trainable = False
        self.first = tf.keras.layers.Dense(BERT_MODEL_OUT_SIZE, activation=activations.softplus)
        self.forth = tf.keras.layers.Dense(1, activation=activations.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        x = self.bert(inputs)
        x = self.first(x)
        if not self.is_work:
            x = self.forth(x)
        return x


class RuBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RU_BERT_DIR)
        self.ru_bert = AutoModel.from_pretrained(RU_BERT_DIR)

    def embed(self, text_inputs):
        t = self.tokenizer(text_inputs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.ru_bert(**{k: v.to(self.ru_bert.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings


def load_work_text_model(checkpoint_dir: str) -> BertBasedModel:
    loaded_model = BertBasedModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def load_name_desc_nn_model(checkpoint_dir: str) -> NameDescModel:
    loaded_model = NameDescModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def load_categorical_nn_model(checkpoint_dir: str) -> CategoricalModel:
    loaded_model = CategoricalModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def load_eval_nn_model(checkpoint_dir: str) -> EvaluateModel:
    loaded_model = EvaluateModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def load_final_model(checkpoint_dir: str) -> FinalModel:
    loaded_model = FinalModel(is_work=True)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    loaded_model.load_weights(latest)
    return loaded_model


def create_name_description_model() -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(8, 8, 1))


def load_eval_model() -> Any:
    with open(EVAL_MODELS_DIR + "/best.pickaim", 'rb') as file:
        return pickle.load(file)


def load_name_desc_model() -> Any:
    with open(NAME_DESC_MODELS_DIR + "/best.pickaim", 'rb') as file:
        return pickle.load(file)


def load_categorical_model() -> Any:
    with open(CATEGORICAL_DIR + "/best.pickaim", 'rb') as file:
        return pickle.load(file)


def load_salary_from_model() -> Any:
    with open(SALARY_FROM_RECOVER_MODELS_DIR + "/best.pickaim", 'rb') as file:
        return pickle.load(file)


def load_final_simple_model() -> Any:
    with open(FINAL_MODELS_DIR + "/best.pickaim", 'rb') as file:
        return pickle.load(file)


def load_ensemble_model() -> Any:
    with open(ENSEMBLE_MODELS_DIR + "/best.pickaim", 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    model = RuBert()
    result = model.embed(['Привет мир'])
    print(result)
