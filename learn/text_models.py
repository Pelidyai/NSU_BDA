import os.path

from pandas import DataFrame
import tensorflow as tf

from models_creation import BertBasedModel
from support.constants import TARGET_NAME, BERT_BASED_NAME_CHECKPOINT_DIR, BERT_BASED_DESCRIPTION_CHECKPOINT_DIR
from support.functions import load_x_prepared_train_data, load_y_train_data


def extract_x_y(data: DataFrame, x_label: str, y_label: str) -> tuple[list, list]:
    return data[x_label].to_numpy(), data[y_label].to_numpy()


def create_and_learn_text_model(data: DataFrame, x_key: str, model_name: str, checkpoints_dir: str) -> tf.keras.Model:
    model = BertBasedModel()
    checkpoint_name = model_name + '-{epoch:04d}.ckpt'
    checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_name)
    model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.MAPE)
    model.save_weights(checkpoint_filepath.format(epoch=0))
    x, y = extract_x_y(data, x_key, TARGET_NAME)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_weights_only=True)
    model.fit(x, y, batch_size=128, epochs=10, shuffle=True, callbacks=[cp_callback])
    return model


def main():
    data = load_x_prepared_train_data()
    data[TARGET_NAME] = load_y_train_data()[TARGET_NAME]
    create_and_learn_text_model(data, 'name', "bert_name_to_5_1", BERT_BASED_NAME_CHECKPOINT_DIR)
    create_and_learn_text_model(data, 'description', "bert_desc_to_5_1", BERT_BASED_DESCRIPTION_CHECKPOINT_DIR)
    # create_and_learn_description_model(data)
    # x, y = extract_x_y(data, 'name', TARGET_NAME)
    # x = None


if __name__ == '__main__':
    main()
    # print(text_clean_up("<strong>Обязанности:</strong> <ul> <li>Ведение работы по формированию сметной "
    #                     "документации;</li> <li>Работа с формами КС-2, КС-3;</li> <li>Анализ стоимости затрат на "
    #                     "строительство, включая стоимость материалов, работу и др.;</li> <li>Защита смет у "
    #                     "заказчика;</li> <li>Составление смет ресурсным методом;</li> <li>Составление смет на "
    #                     "ПИР;</li> <li>Расчет и проверка объемов работы по чертежам;</li> <li>Ведение накопительных "
    #                     "ведомостей.</li> </ul> <strong>Требования:</strong> <ul> <li>Высшее образование ("
    #                     "строительное, желательно ПГС);</li> <li>Опыт работы сметчиком от 1-3 лет;</li> <li>Свободное"
    #                     " чтение чертежей, проектной документации;</li> <li>Знание строительства;</li> <li>Уверенный "
    #                     "пользователь компьютера (Smeta.ru, MS Word, MS Excel);</li> <li>Грамотность, "
    #                     "ответственность, исполнительность, умение работать в многозадачном режиме;</li> </ul> "
    #                     "<strong>Условия:</strong> <ul> <li>Оформление согласно ТК РФ;</li> <li>График: 5/2 П-Ч с "
    #                     "8:00-17:00, П до 16:45</li> <li>Дружный коллектив;</li> <li>Комфортный офис, "
    #                     "м. Калужская;</li> <li>Полный рабочий день;</li> <li>Резюме просматривается только с "
    #                     "фотографией;</li> <li>На территории работодателя.</li> </ul>"))
