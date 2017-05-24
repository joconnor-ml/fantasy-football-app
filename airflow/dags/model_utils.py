import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
from sklearn_pandas import DataFrameMapper
from bayesian_models import BayesianPointsRegressor, MeanPointsRegressor

import logging

def get_data(test_week, one_hot):
    df = pd.read_csv("data.csv")
    df.index = df.web_name
    df.to_csv("test_details.csv")
    if one_hot:
        opponent_team = pd.get_dummies(df["target_team"].fillna(999).astype(int)).add_prefix("opponent_")
        own_team = pd.get_dummies(df["team_code"].fillna(999).astype(int)).add_prefix("team_")
        position = pd.get_dummies(df["element_type"].fillna(999).astype(int)).add_prefix("position_")
        df = pd.concat([
            df.drop(["target_team", "team_code", "element_type"], axis=1),
            opponent_team, own_team, position
        ], axis=1)
    df = df[df["target_minutes"] > 60]
    y = df["target"]
    X = df.drop(["target", "id", "target_minutes", "web_name"], axis=1).astype(np.float64)
    notnull = y.notnull()
    X = X[notnull]
    y = y[notnull]
    train = X["gameweek"] < test_week
    test = X["gameweek"] == test_week
    return X[train], X[test], y[train], y[test]

models = {
    "xgb":
    XGBRegressor(n_estimators=64, learning_rate=0.1, max_depth=1),
    "xgb2":
    XGBRegressor(n_estimators=64, learning_rate=0.1, max_depth=2),
    "rf":
    make_pipeline(Imputer(), RandomForestRegressor(n_estimators=100, max_depth=3)),
    "linear":
    make_pipeline(
        Imputer(), MinMaxScaler(), RidgeCV(),
    ),
    "polynomial":
    make_pipeline(
        Imputer(), PolynomialFeatures(), MinMaxScaler(), PCA(16),  RidgeCV(),
    ),
    "simple_mean":
    MeanPointsRegressor(),
    "bayes_global_prior":
    BayesianPointsRegressor("global"),
    "bayes_team_prior":
    BayesianPointsRegressor("team"),
    "bayes_position_prior":
    BayesianPointsRegressor("position"),
}
