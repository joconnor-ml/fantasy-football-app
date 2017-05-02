"""Get data into nice form for training
our models"""

from os.path import dirname, join

import numpy as np
import pandas as pd
import pymongo

client = pymongo.MongoClient()
db = client.fantasy_football
# TODO only grab latest data. currently duplicating.
player_history = db.player_data.find({},{"history":1})
player_dfs = {}
for i, player in enumerate(player_history):
    if i < 649:
        continue
    df = pd.DataFrame(player["history"])
    df.index = df["round"]
    player_dfs[i] = df

player_panel = pd.Panel(player_dfs)
player_panel.loc[:, :, "appearances"] = player_panel.loc[:, :, "minutes"] > 0
player_panel.loc[:, :, "none"] = 1

player_details = pd.DataFrame(db.player_details.find_one({},{"elements":1})["elements"])
player_details.index = player_details.id
