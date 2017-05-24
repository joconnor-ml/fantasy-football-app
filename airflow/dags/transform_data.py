"""Get data into nice form for training
our models"""

from os.path import dirname, join

import numpy as np
import pandas as pd
import pymongo

def transform_data(execution_date, **kwargs):
    client = pymongo.MongoClient()
    db = client["fantasy_football"]
    
    player_details = pd.DataFrame(list(db["elements"].find()))
    player_details.index = player_details.id
    player_details = player_details[["team_code", "web_name", "element_type"]]

    player_history = db["player_data"].find({},{"history":1})
    player_dfs = {}
    for i, player in enumerate(player_history):
        df = pd.DataFrame(player["history"])
        df = df.reset_index()
        df.index += df["round"].min()
        player_df = df[["minutes", "total_points", "was_home", "opponent_team"]].astype(np.float64)
        player_df.loc[:, "appearances"] = (player_df.loc[:, "minutes"] > 0).astype(np.float64)
        mean3 = player_df[["total_points", "minutes", "appearances"]].rolling(3).mean()
        mean10 = player_df[["total_points", "minutes", "appearances"]].rolling(10).mean()
        std10 = player_df[["total_points", "minutes", "appearances"]].rolling(10).std()
        mean5 = player_df[["total_points", "minutes", "appearances"]].rolling(5).mean()
        ewma = player_df[["total_points", "minutes", "appearances"]].ewm(halflife=10).mean()
        cumulative_sums = player_df.cumsum(axis=0)
        # normalise by number of games played up to now
        cumulative_means = cumulative_sums[["total_points"]].div(cumulative_sums.loc[:, "appearances"] + 1, axis=0)
        player_df["id"] = i + 1
        # join on player details to get position ID, name and team ID.
        player_df = pd.merge(player_df, player_details,
                             how="left", left_on="id", right_index=True)
        player_df["target"] = player_df["total_points"].shift(-1)
        player_df["target_minutes"] = player_df["minutes"].shift(-1)
        player_df["target_home"] = player_df["was_home"].shift(-1)
        player_df["target_team"] = player_df["opponent_team"].shift(-1)
        player_df["gameweek"] = player_df.index

        weight = cumulative_sums["appearances"] / (cumulative_sums["appearances"] + 30)
        player_df["bayes_global"] = cumulative_means["total_points"] * weight + 3.5 * (1 - weight)

        #one_hot = True
        
        #if one_hot:
        # apply one-hot encoding to categorical variables
        #opponent_team = pd.get_dummies(player_df["target_team"]).add_prefix("opponent_")
        #own_team = pd.get_dummies(player_df["team_code"]).add_prefix("team_")
        #position = pd.get_dummies(player_df["element_type"]).add_prefix("position_")
        #player_df = pd.concat([player_df.drop(["target_team", "team_code", "element_type"], axis=1),
        #                           opponent_team, own_team, position], axis=1)

        player_dfs[i] = pd.concat([
            player_df,
            mean3.add_suffix("_mean3"),
            mean5.add_suffix("_mean5"),
            mean10.add_suffix("_mean10"),
            std10.add_suffix("_std10"),
            ewma.add_suffix("_ewma"),
            cumulative_means.add_suffix("_mean_all"),
            cumulative_sums.add_suffix("_sum_all"),
        ], axis=1)
        
    player_df = pd.concat(player_dfs)
    player_df.to_csv("data.csv")
    #player_details.to_csv("player_details.csv")  # store the player names and such so we can inspect them during training/validation

if __name__ == "__main__":
    transform_data()
