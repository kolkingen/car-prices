"""
Скрипт загружает данные из data/test, делает для них 
предсказания и сохраняет в папку data/predictions. 
Используется в качестве оператора airflow.
"""


from os.path import join as join_path
import glob
import json
import logging
from datetime import datetime

import dill
import pandas as pd


def predict() -> None:

    # Загрузим последнюю сохранённую модель
    models_path = join_path('data', 'models')
    model_files = glob.glob(join_path(models_path, 'cars_pipe_*.pkl'))
    last_model = sorted(model_files)[-1]
    with open(last_model, 'rb') as file:
        model = dill.load(file)

    logging.info(f'Модель успешно загружена, {last_model.split("/")[-1]}')

    # Загрузим данные для предсказания
    test_files = glob.glob(join_path('data', 'test', '*.json'))

    test_data = list()
    for f in test_files: 
        with open(f, 'r') as file:
            test_data.append(json.load(file))

    logging.info('Данные для теста успешно загружены')

    # Проведем предсказание
    test_data = pd.DataFrame(test_data)
    test_data['prediction'] = model['model'].predict(test_data)

    # Сохраним предсказания
    predictions_filename = f'preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predictions_path = join_path('data', 'predictions', predictions_filename)
    test_data[['id', 'prediction']].to_csv(predictions_path, index=False)

    logging.info(f'Предсказания сохранены в файл {predictions_filename} в data/predictions')


if __name__ == '__main__':
    predict()
