import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator


# Укажем путь к файлам проекта:
project_path = os.path.join(os.path.expanduser('~'), 'Projects', 'ds-car-prices')

# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, project_path)


from create_model import create_model
from predict import predict


args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:

    # Создание модели 
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=create_model,
    )

    # Предсказание на тестовых данных
    predict = PythonOperator(
        task_id='predict', 
        python_callable=predict, 
    )

    # Зададим порядок выполнения
    pipeline >> predict
