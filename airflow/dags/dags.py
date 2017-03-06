import logging
from airflow.operators import PythonOperator
from airflow.models import DAG
from datetime import datetime, timedelta


def download_data(**kwargs):
    logging.info("download_data")
    return []
    
args = {
    'owner': 'airflow',
    'start_date': datetime(2017,2,5),
}


dag = DAG(
    dag_id='data_import',
    default_args=args,
    schedule_interval=timedelta(days=1),
)


download_data = PythonOperator(
    task_id='download_data',
    provide_context=True,
    python_callable=download_data,
    dag=dag,
)


