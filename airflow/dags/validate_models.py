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


def validate_model(model, model_name):
    pred_list = []
    ys = []
    scores = []
    for test_week in range(30, 37):
        if model_name == "linear":
            Xtrain, Xtest, ytrain, ytest = model_utils.get_data(test_week=test_week,
                                                                one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest = model_utils.get_data(test_week=test_week,
                                                                one_hot=False)
            
        preds = model.fit((Xtrain), ytrain).predict((Xtest))
        rmse = mean_squared_error(ytest, preds) ** 0.5
        imps = None
        if "xgb" in model_name:
            imps = pd.Series(model.booster().get_fscore())
        elif model_name == "linear":
            imps = pd.Series(model.steps[-1][-1].coef_,
                             index=Xtrain.columns).abs()
        #if imps is not None:
        #    logging.info("\n{}".format(imps.sort_values().tail()))
        #    logging.info("RMSE {}: {}".format(model_name, rmse))
        pred_list.append(preds)
        ys.append(ytest)
        scores.append(mean_squared_error(ytest, preds) ** 0.5)

    ytest = np.concatenate(ys)
    preds = np.concatenate(pred_list)
    rmse = mean_squared_error(ytest, preds) ** 0.5
    logging.info(scores)
    logging.info("RMSE Combo: {}".format(rmse))

    
def validate_models(execution_date, **kwargs):
    for name, model in model_utils.models.items():
        logging.info(name)
        validate_model(model, name)
        

if __name__ == "__main__":
    import datetime
    validate_models(datetime.datetime.now())
