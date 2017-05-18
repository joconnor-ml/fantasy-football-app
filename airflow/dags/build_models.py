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

import logging


def build_models(execution_date, **kwargs):
    details = pd.read_csv("player_details.csv", index_col=0, nrows=650)[["team_code", "web_name"]]
    details.index = details.index.astype(np.float64)

    test_week = 30
    panel = pd.read_pickle("data.pkl").swapaxes(0,2)
    # pick a week to test on:
    test = panel.loc[:, test_week, :]
    test.index = details.web_name
    test = pd.merge(test, details.select_dtypes(["number"]), how="left", left_on="id", right_index=True)
    test["team_code"] = test["team_code"].fillna(999)
    test = test[test["target_minutes"] > 60]
    ytest = test["target"]
    Xtest = test.drop(["target", "id", "target_minutes"], axis=1).astype(np.float64)
    #logging.info("\n{}".format(test.head()))

    train = panel.loc[:, 10:test_week-2, :].to_frame()  # flatten
    train = pd.merge(train, details.select_dtypes(["number"]), how="left", left_on="id", right_index=True)
    train["team_code"] = train["team_code"].fillna(99)
    train = train[train["target_minutes"]>60]
    ytrain = train["target"]
    Xtrain = train.drop(["target", "id", "target_minutes"], axis=1).astype(np.float64)

    model = XGBRegressor(n_estimators=64, learning_rate=0.1, max_depth=2)
    model.fit((Xtrain), ytrain)
    xgb_preds = model.predict((Xtest))
    notnans = ytest.notnull()
    rmse = mean_squared_error(ytest[notnans], xgb_preds[notnans]) ** 0.5
    imps = pd.Series(model.booster().get_fscore())
    logging.info("\n{}".format(imps.sort_values().tail()))
    logging.info("")
    xgb_pred_df = pd.DataFrame({"xgb_preds": xgb_preds, "score":ytest.values}, index=test.index)
    logging.info("\n{}".format(xgb_pred_df.sort_values("xgb_preds").dropna().tail()))
    logging.info("RMSE XGB: {}".format(rmse))
    logging.info("")

    mapper = DataFrameMapper([
        (["team_code"], OneHotEncoder(sparse=False)),
    ], default=None)
    #union = make_union(FunctionTransformer(lambda x: x.drop("team_code", axis=1)),
    #                   make_pipeline(FunctionTransformer(lambda x: x["team_code"]), OneHotEncoder()))
    model = make_pipeline(mapper, Imputer(), PolynomialFeatures(),
                          MinMaxScaler(), RidgeCV())
    model.fit((Xtrain), ytrain)
    lr_preds = model.predict((Xtest))
    notnans = ytest.notnull()
    rmse = mean_squared_error(ytest[notnans], lr_preds[notnans]) ** 0.5
    #imps = pd.Series(model.steps[-1][-1].coef_, index=Xtrain.columns).abs()
    #logging.info("\n{}".format(imps.sort_values().tail()))
    #logging.info("")
    lr_pred_df = pd.DataFrame({"lr_preds": lr_preds, "score":ytest.values}, index=test.index)
    logging.info("\n{}".format(lr_pred_df.sort_values("lr_preds").dropna().tail()))
    logging.info("RMSE LR: {}".format(rmse))
    logging.info("")

    combo_preds = 0.5*(lr_preds + xgb_preds)
    pred_df = pd.DataFrame({"combo_preds": combo_preds, "score":ytest.values}, index=test.index)
    logging.info("\n{}".format(pred_df.sort_values("combo_preds").dropna().head()))
    logging.info("\n{}".format(pred_df.sort_values("combo_preds").dropna().tail()))
    rmse = mean_squared_error(ytest[notnans], combo_preds[notnans]) ** 0.5
    logging.info("RMSE Combo: {}".format(rmse))
    

if __name__ == "__main__":
    build_models()
