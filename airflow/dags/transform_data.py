"""Get data into nice form for training
our models"""

from os.path import dirname, join

import numpy as np
import pandas as pd
import pymongo

def transform_data():
    client = pymongo.MongoClient()
    db = client["fantasy_football"]
    
    player_history = db["player_data"].find({},{"history":1})
    player_dfs = {}
    for i, player in enumerate(player_history):
        df = pd.DataFrame(player["history"])
        df = df.reset_index()
        df.index += df["round"].min()
        player_dfs[i] = df.select_dtypes(["number"])
        
    player_panel = pd.Panel(player_dfs)
    player_panel.loc[:, :, "appearances"] = player_panel.loc[:, :, "minutes"] > 0
    player_panel.loc[:, :, "none"] = 1
        
    player_details = pd.DataFrame(list(db["elements"].find()))
    player_details.index = player_details.id
    
    player_panel.loc[:, :, "target"] = player_panel.loc[:, :, "total_points"].shift(-1)
    cumulative_sums = player_panel.cumsum(axis=1)
    cumulative_means = cumulative_sums.div(cumulative_sums.loc[:, :, "appearances"]+1e-6, axis=1)

    cumulative_means = cumulative_means.add_suffix("_mean")
    pd.concat([cumulative_means, player_panel], axis=2).to_pickle("data.pkl")
    
if __name__ == "__main__":
    transform_data()
