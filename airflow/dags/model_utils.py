import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline, make_union
from xgboost import XGBRegressor
from sklearn_pandas import DataFrameMapper

import logging

def prepare(df):
    details = pd.read_csv("player_details.csv",
                          index_col=0)[["team_code", "web_name", "element_type"]]
    details.index = details.index.astype(np.float64)
    df = pd.merge(df, details,
                  how="left", left_on="id", right_index=True)
    df.index = df.web_name
    df.to_csv("test_details.csv")
    df["team_code"] = df["team_code"].fillna(999)
    df["target_team"] = df["target_team"].fillna(999)
    df["element_type"] = df["element_type"].fillna(999)
    df = df[df["target_minutes"] > 60]
    y = df["target"]
    X = df.drop(["target", "id", "target_minutes", "web_name"], axis=1).astype(np.float64)
    notnull = y.notnull()
    return X[notnull], y[notnull]

models = {
    "xgb":
    XGBRegressor(n_estimators=64, learning_rate=0.1, max_depth=1),
    "linear":
    make_pipeline(
        DataFrameMapper([
            (["team_code", "target_team", "element_type"],
             OneHotEncoder(sparse=False)),
        ], default=None),
        Imputer(), MinMaxScaler(), RidgeCV(),
    )
}
