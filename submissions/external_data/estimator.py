from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(data_train):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    
    data_train = data_train.copy()
    # When using merge_asof left frame need to be sorted
    data_train['orig_index'] = np.arange(data_train.shape[0])
    data_train = pd.merge_asof(data_train.sort_values('date'), df_ext[['date', 'dd', 'ff', 't', 'td', 'u', 'vv', 'total_cases', 'new_deaths', 'is_workday', 'lockdown', 'curfew']].sort_values('date'), on='date')
    # Sort back to the original order
    data_train = data_train.sort_values('orig_index')
    del data_train['orig_index']
    return data_train



def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month', 'day', 'weekday', 'hour']

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    numeric_cols = ['dd', 'ff', 't', 'td', 'u', 'vv', 'total_cases', 'new_deaths', 'is_workday', 'lockdown', 'curfew']

    preprocessor = ColumnTransformer([
        ('date', "passthrough", date_cols),
        ('cat', categorical_encoder, categorical_cols),
        ('numeric', 'passthrough', numeric_cols)
    ])

    regressor = xgb.XGBRegressor(learning_rate=0.1, n_estimators=900, max_depth=8, min_child_weight=2, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
            nthread=4, scale_pos_weight=1, seed=27)

    pipe =  make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor
    )

    return pipe
