"""
Скрипт работает как сервис для предсказания категории машин по их признакам. 
Используется фреймфорк FastAPI.

Модель загружается из файла `cars_pipe.pkl`, 
который был заранее создан скриптом `create_model.py`.
"""


from os.path import join as join_path
import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# создадим объект для работы с FastAPI
app = FastAPI()

# загрузим модель и метаданные с помощью dill
with open(join_path('data', 'models', 'cars_pipe.pkl'), 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    """ Структура данных для функции predict. """

    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


class Prediction(BaseModel):
    """ Структуа данных для ответа функции predict. """

    id: int
    price: int
    prediction: str


@app.get('/status')
def status() -> str:
    return "I'm OK"


@app.get('/version')
def version() -> dict:
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form) -> dict:

    df = pd.DataFrame([form.dict()])
    y = model['model'].predict(df)

    return {'id': form.id, 'price': form.price, 'prediction': y[0]}


if __name__ == '__main__':

    # для проверки функции predict
    sample = {
        "description": "Excellent condition with Leather Stone interior.  5.0 V-8 engine. Garage kept. All maintenance preformed by Bob Bell Ford. Comes with Premium Extended Service Plan until April 2022. Also comes with Premium Maintenance Plan until 2022. Six speed manual transmission with \"Hill Assist\". Comes with Carpeted floor mats, but also with Rubber mats for winter, spare donut in trunk, but comes with spare full size WHEEL ONLY.  Sirius radio capable, back up camera, back up sensors, power seats, heated seats, LED lights.",
        "fuel": "gas",
        "id": 7316509996,
        "image_url": "https://images.craigslist.org/00808_i0faaALGPQxz_0CI0t2_600x450.jpg",
        "lat": 39.1618,
        "long": -76.6297,
        "manufacturer": "ford",
        "model": "mustang",
        "odometer": 53800.0,
        "posting_date": "2021-05-03T19:49:21-0400",
        "price": 24500,
        "region": "baltimore",
        "region_url": "https://baltimore.craigslist.org",
        "state": "md",
        "title_status": "clean",
        "transmission": "manual",
        "url": "https://baltimore.craigslist.org/cto/d/glen-burnie-mustang-50-convertible-2013/7316509996.html",
        "year": 2013.0
    }
    sample_form = Form.parse_obj(sample)
    print(predict(sample_form))
