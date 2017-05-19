"""Get data into nice form for training
our models"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline, make_union
from xgboost import XGBRegressor
from sklearn_pandas import DataFrameMapper

import model_utils

import logging

def prepare(df):
    df = pd.merge(df, details.select_dtypes(["number"]), how="left", left_on="id", right_index=True)
    df["team_code"] = df["team_code"].fillna(999)
    df["target_team"] = df["target_team"].fillna(999)
    df["element_type"] = df["element_type"].fillna(999)
    df = df[df["target_minutes"] > 60]
    y = df["target"]
    X = df.drop(["target", "id", "target_minutes"], axis=1).astype(np.float64)
    return X, y

    
def build_models(execution_date, **kwargs):
    test_week = 30
    panel = pd.read_pickle("data.pkl").swapaxes(0,2)
    # pick a week to test on:
    test = panel.loc[:, test_week, :]
    Xtest, ytest = model_utils.prepare(test)
    # train on all previous weeks
    train = panel.loc[:, 10:test_week-2, :].to_frame()  # flatten
    Xtrain, ytrain = model_utils.prepare(train)

    pred_list = []
    for name, model in model_utils.models.items():
        preds = model.fit((Xtrain), ytrain).predict((Xtest))
        rmse = mean_squared_error(ytest, preds) ** 0.5
        if name == "xgb":
            imps = pd.Series(model.booster().get_fscore())
            logging.info("\n{}".format(imps.sort_values().tail()))
            logging.info("")
        #elif name == "linear":
        #    imps = pd.Series(model.steps[-1][-1].coef_,
        #                     index=Xtrain.columns).abs()
        logging.info("RMSE {}: {}".format(name, rmse))
        logging.info("")
        pred_list.append(preds)

    combo = sum(pred_list)/len(pred_list)
    pred_df = pd.DataFrame({"preds": combo_preds, "score":ytest.values},
                           index=test.index)
    logging.info(
        "\n{}".format(pred_df.sort_values("preds").dropna().head())
    )
    
    rmse = mean_squared_error(ytest, sum(pred_list)/len(pred_list)) ** 0.5
    logging.info("RMSE Combo: {}".format(rmse))
    

if __name__ == "__main__":
    import datetime
    build_models(datetime.datetime.now())
