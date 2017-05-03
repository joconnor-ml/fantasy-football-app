"""Get data into nice form for training
1;3409;0cour models"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


def build_models():
    details = pd.read_csv("player_details.csv", index_col=0, nrows=650)

    test_week = 21
    panel = pd.read_pickle("data.pkl").swapaxes(0,2)
    # pick a week to test on:
    test = panel.loc[:, test_week, :]
    ytest = test["target"]
    Xtest = test.drop(["target"], axis=1).astype(np.float32)
    print(test.head())

    train = panel.loc[:, 10:test_week, :].to_frame()  # flatten
    ytrain = train["target"]
    Xtrain = train.drop(["target"], axis=1).astype(np.float32)
    
    model = XGBRegressor(n_estimators=32, learning_rate=0.2, max_depth=8)
    model.fit((Xtrain), ytrain)
    preds = model.predict((Xtest))
    notnans = ytest.notnull()
    rmse = mean_squared_error(ytest[notnans], preds[notnans]) ** 0.5
    imps = pd.Series(model.booster().get_fscore())
    print(imps.sort_values().tail(10))
    print()
    preds = pd.DataFrame({"preds": preds, "score":ytest.values}, index=details.web_name)
    print(preds.sort_values("preds").tail())
    print(rmse)
    print()

    model = make_pipeline(Imputer(), MinMaxScaler(), LinearRegression())
    model.fit((Xtrain), ytrain)
    preds = model.predict((Xtest))
    notnans = ytest.notnull()
    rmse = mean_squared_error(ytest[notnans], preds[notnans]) ** 0.5
    imps = pd.Series(model.steps[-1][-1].coef_, index=Xtrain.columns).abs()
    print(imps.sort_values().tail(10))
    print()
    preds = pd.DataFrame({"preds": preds, "score":ytest.values}, index=details.web_name)
    print(preds.sort_values("preds").tail())
    print(rmse)
    print()



                     
if __name__ == "__main__":
    build_models()
