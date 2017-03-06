"""download fpl datafrom web api to file.
Could stick them in a database? Might be a nice use of MongoDB.
Or is this somewhere to use Kafka?"""
from datetime import datetime
import requests
import json
import time

from pymongo import MongoClient

PLAYER_URL = "https://fantasy.premierleague.com/drf/element-summary/"
PLAYER_DETAIL_URL = "https://fantasy.premierleague.com/drf/bootstrap-static"
MAX_PLAYER_ID = 650


client = MongoClient()
db = client["fantasy_football"]


def download_data(execution_date, schedule_interval):
    response = requests.get(PLAYER_DETAIL_URL)
    data = response.json()
    data["retrieval_date"] = datetime.now().isoformat()
    
    collection = db["player_details"]
    collection.insert_one(data)

    print("got player details")
    for player_id in range(1, MAX_PLAYER_ID+1):
        if player_id%100==0:
            print(player_id)
        filename = "dumps/fpl.player_"
        "{}.{}.{}.{}.json".format(player_id,
                                  execution_date.year,
                                  execution_date.month,
                                  execution_date.day)
        print("ID", player_id)
        response = requests.get("{}/{}".format(PLAYER_URL, player_id))
        data = response.json()
        data["player_id"] = player_id
        data["retrieval_date"] = datetime.now().isoformat()
        collection = db["player_data"]
        collection.insert_one(data)
        time.sleep(10)


if __name__ == "__main__":
    from datetime import datetime
    download_data(datetime.today(), None)
