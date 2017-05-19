"""Get data into nice form for training
our models"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline, make_union
from xgboost import XGBRegressor
from sklearn_pandas import DataFrameMapper

import model_utils

import logging

    
def validate_models(execution_date, **kwargs):
    test_week = 30
    panel = pd.read_pickle("data.pkl").swapaxes(0,2)
    # pick a week to test on:
    test = panel.loc[:, test_week, :]
    Xtest, ytest = model_utils.prepare(test)
    # train on all previous weeks
    train = panel.loc[:, 10:test_week-2, :].to_frame()  # flatten
    Xtrain, ytrain = model_utils.prepare(train)

    pd.concat([Xtest, ytest], axis=1).to_csv("xtest.csv")

    best_constant = ytrain.mean()
    logging.info("Best constant: {}".format(best_constant))
    
    # use mean of ytrain to predict ytest
    rmse = mean_squared_error(ytest, np.ones_like(ytest)*best_constant) ** 0.5
    rmse = mean_absolute_error(ytest, np.ones_like(ytest)*best_constant) ** 1
    logging.info("RMSE constant bechmark: {}".format(rmse))

    # use prior points per game to predict ytest
    rmse = mean_squared_error(ytest, 0.5*(Xtest["total_points_mean_all"] + np.ones_like(ytest)*best_constant)) ** 0.5
    rmse = mean_absolute_error(ytest, 0.5*(Xtest["total_points_mean_all"] + np.ones_like(ytest)*best_constant)) ** 1
    logging.info("RMSE points-per-game bechmark: {}".format(rmse))

    pred_list = []
    for name, model in model_utils.models.items():
        preds = model.fit((Xtrain), ytrain).predict((Xtest))
        rmse = mean_squared_error(ytest, preds) ** 0.5
        rmse = mean_absolute_error(ytest, preds) ** 1
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
    pred_df = pd.DataFrame({"preds": combo, "score":ytest}, index=ytest.index)
    logging.info(
        "\n{}".format(pred_df.sort_values("preds").dropna().tail())
    )
    
    rmse = mean_squared_error(ytest, combo) ** 0.5
    rmse = mean_absolute_error(ytest, combo) ** 1
    logging.info("RMSE Combo: {}".format(rmse))


if __name__ == "__main__":
    import datetime
    validate_models(datetime.datetime.now())
