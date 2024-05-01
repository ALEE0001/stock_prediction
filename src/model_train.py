import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
    KBinsDiscretizer,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)

import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier

import torch
import torch.nn as nn

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)


class PrepData:
    """
    Description:
        prepares data for model training.

    Attributes:
        df (dataframe):
        date_col (str): Column Name for date in yyyy-mm-dd format
        ticker_col (str): Column Name for ticker/symbol
        close_col (str): Column Name for close price
        target_type (str): [7dDirection, 7dPercReturn]

    Methods:
        find_value_7_days_later(self, date)
        generate_target(self)
        create_train_test(self)
    """

    def __init__(
        self,
        df,
        date_col,
        ticker_col,
        close_col,
        targ_direction=False,
        targ_perc_return=False,
    ):
        self.df = df
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.close_col = close_col
        self.targ_direction = targ_direction
        self.targ_perc_return = targ_perc_return

    def find_value_7_days_later(self, date):
        next_date = date + pd.Timedelta(days=7)
        next_row = self.df.loc[self.df[self.date_col] == next_date]
        if not next_row.empty:
            return next_row.iloc[0][self.close_col]
        else:
            return None

    def generate_target(self):
        self.df["7dClose"] = self.df.groupby(self.ticker_col)[self.date_col].apply(
            lambda group: group.apply(self.find_value_7_days_later)
        )
        if self.targ_direction:
            self.df["target"] = np.where(
                self.df[self.close_col] > self.df["7dClose"], 1, 0
            )

        elif self.targ_perc_return:
            self.df["target"] = (
                self.df["7dClose"] - self.df[self.close_col]
            ) / self.df["7dClose"]

        else:
            raise ValueError("One of [targ_direction, targ_perc_return] has to be True")

        self.df.drop("7dClose", axis=1, inplace=True)

    def create_x_y(self):
        self.generate_target()

        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        return X, y


class CreateLGBMModel:
    """
    Description:
        Builds Model

    Attributes:
        X (dataframe): features
        y (array): target label
        task (str): [regression, classification]

    Methods:
        create_lgbm_pipeline(self)
        create_lstm_pipeline(self)
    """

    def __init__(self, X, y, features, task):
        self.X = X
        self.y = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.X["GICS Sector"]
        )

        self.features = features
        self.task = task

    def create_pipeline(self):

        misc_cols = [c for c in self.X.columns if c not in self.features]

        # X.drop(drop_cols, axis=1, inplace=True)

        # Identify numerical and categorical columns
        numerical_cols = self.X[self.features].select_dtypes(include=["number"]).columns
        categorical_cols = (
            self.X[self.features].select_dtypes(exclude=["number"]).columns
        )

        # Define preprocessing steps for numerical and categorical columns
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        # Combine preprocessing steps using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Create the pipeline with preprocessor and LightGBM Regressor
        if self.task == "regression":
            model = Pipeline(
                steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor())]
            )
        elif self.task == "classification":
            model = Pipeline(
                steps=[("preprocessor", preprocessor), ("regressor", LGBMClassifier())]
            )

        return model

    def hpo(self):

        model = self.create_pipeline()

        opt = BayesSearchCV(
            model,
            {
                # 'boosting_type': 'gbdt',
                "num_leaves": Integer(10, 100),
                # 'max_depth':-1,
                "learning_rate": Real(0.0001, 0.1),
                "n_estimators": Integer(100, 5000),
                # 'subsample_for_bin': 200000,
                # 'objective': None,
                # 'class_weight': None,
                "min_split_gain": Real(0.0, 0.1),
                "min_child_weight": Real(0.001, 0.01),
                "min_child_samples": Integer(20, 10000),
                # 'subsample': 1.0,
                # 'subsample_freq': 0,
                # 'colsample_bytree': 1.0,
                "reg_alpha": Real(1, 80),
                "reg_lambda": Real(1, 100),
                # 'random_state': None,
                # 'n_jobs': None,
                "importance_type": Categorical(["gain"]),
            },
            n_iter=50,
            random_state=42,
        )

        opt.fit(self.X_train, self.y_train)

        print("best params: %s" % str(opt.best_params_))
