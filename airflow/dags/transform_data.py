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
        player_dfs[i] = df.select_dtypes(["number"]).astype(np.float32).drop(["loaned_in", "loaned_out", "ea_index",
                                                                              "index", "element",
                                                                              "transfers_in", "transfers_out", "transfers_balance"], axis=1)
        
    player_panel = pd.Panel(player_dfs)
    player_panel.loc[:, :, "appearances"] = player_panel.loc[:, :, "minutes"] > 0
    player_panel.loc[:, :, "none"] = 1
    print(player_panel.loc[201, 10, :])
        
    player_details = pd.DataFrame(list(db["elements"].find()))
    player_details.index = player_details.id
    
    # for each gameweek, get the total of each attribute up to now
    cumulative_sums = player_panel.cumsum(axis=1)
    # normalise by number of games played up to now
    cumulative_means = cumulative_sums.div(cumulative_sums.loc[:, :, "appearances"]+1, axis=2)

    name_map = {name: name + "_mean" for name in cumulative_means.axes[2]}
    cumulative_means = cumulative_means.rename(minor_axis=name_map)
    panel = pd.concat([cumulative_means, cumulative_sums], axis=2)
    panel.loc[:, :, "target"] = sum(player_panel.loc[:, :, "total_points"].shift(-i) for i in range(1, 5))
    panel.to_pickle("data.pkl")
    player_details.to_csv("player_details.csv")  # store the player names and such so we can inspect them during training/validation
    
if __name__ == "__main__":
    transform_data()
