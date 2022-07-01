"""
Скрипт тренирует несколько моделей, используя механизм пайплайнов sklearn.
Лучшая модель по результатам обучения сохраняется в файл `cars_pipe.pkl`, 
после чего используется в запущенном сервисе (см. файл `main.py`).
"""


from os.path import join as join_path
from datetime import datetime
import logging

import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def filter_data(data):
    """ Удаление ненужных колонок из датасета. """

    columns_to_drop = ['id', 'url', 'region', 'region_url', 'price', 'manufacturer',
                       'image_url', 'description', 'posting_date', 'lat', 'long']

    return data.drop(columns_to_drop, axis=1)


def remove_year_outliers(data):
    """ Сглаживание выбросов в колонке year. """

    # создадим копию датафрейма
    data = data.copy()

    # рассчитаем границы для интерквантильного размаха
    q25 = data['year'].quantile(0.25)
    q75 = data['year'].quantile(0.75)
    boundaries = (q25 - 1.5 * (q75 - q25), q75 + 1.5 * (q75 - q25))

    # удалим выбросы
    data.loc[data['year'] < boundaries[0], 'year'] = round(boundaries[0])
    data.loc[data['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return data


def new_features(data):
    """ Создание новых признаков: short_model и age_category. """

    # скопируем датасет
    data = data.copy()

    # создадим признак short_model - первое слово из колонки model
    data['short_model'] = data['model'].apply(lambda x: str(x).lower().split()[0])

    # создадим признак age_category - категорию возраста авто
    data['age_category'] = data['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

    return data


def create_model():

    # загрузим данные и отделим целевую переменную
    df = pd.read_csv(join_path('data', 'train', 'train.csv'))
    x = df.drop('price_category', axis=1)
    y = df['price_category']

    logging.info(f'Data is loaded.')

    # создадим конвейер для обработки численных признаков
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # создадим конвейер для обработки категориальных признаков
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore'))
    ])

    # объединим эти два конвейера
    num_and_cat_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])

    # создадим конвейер для предварительной обработки данных
    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('data_cleaning', FunctionTransformer(remove_year_outliers)),
        ('feature_engineering', FunctionTransformer(new_features)),
        ('num_and_cat_transformer', num_and_cat_transformer)
    ])

    # создадим пайплайны для рассматриваемых моделей
    # и выберем модель с лучшим результатом
    best_score, best_pipe = 0.0, None
    models = [LogisticRegression(solver='liblinear'), RandomForestClassifier(), SVC()]

    for model in models:
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        score = cross_val_score(pipe, x, y, cv=4, scoring='accuracy')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
        logging.info(f'model: {type(model).__name__}, acc_mean: {score.mean():0.4f}, acc_std: {score.std():0.4f}.')

    # обучим лучшую модель на всём датасете
    logging.info(f'\nbest_model: {type(best_pipe.named_steps["classifier"]).__name__}, acc_mean: {best_score:0.4f}')
    best_pipe.fit(x, y)

    # сохраним модель вместе с метаданными
    data_to_save = {
        'model': best_pipe,
        'metadata': {
            'name': 'cars\' price category prediction pipeline',
            'author': 'Nikolai Borziak',
            'version': 1.0,
            'date': datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'accuracy': best_score
        }
    }

    file_name = f'cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'
    file_path = join_path('data', 'models', file_name)
    with open(file_path, 'wb') as file:
        dill.dump(data_to_save, file)
    
    logging.info(f'Model is saved as {file_name} in data/models')


if __name__ == '__main__':
    create_model()
