import logging
from airflow.operators import PythonOperator
from airflow.models import DAG
from datetime import datetime, timedelta
from download_data import download_data


def make_task(func):
    return PythonOperator(
        task_id=func.__name__,
        provide_context=True,
        python_callable=func,
        dag=dag,
    )

    
args = {
    'owner': 'airflow',
    'start_date': datetime(2017,4,28,0,0,0),
}

dag = DAG(
    dag_id='fantasy_football',
    default_args=args,
    schedule_interval=timedelta(weeks=1),
)


def transform_data():
    pass


def build_models():
    pass


def produce_predictions():
    pass


# define tasks
import_task = make_task(download_data)
transform_task = make_task(transform_data)
model_task = make_task(build_models)
predict_task = make_task(produce_predictions)

# define dependencies
transform_task.set_upstream(import_task)
model_task.set_upstream(transform_task)
predict_task.set_upstream(model_task)
