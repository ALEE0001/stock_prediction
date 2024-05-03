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


class ModelPrepData:
    """
    Description:
        prepares data for model training.

    Attributes:
        df (dataframe):
        ticker_col (str): Column Name for ticker/symbol
        close_col (str): Column Name for close price
        features (list): List of feature column names
        target_type (str): [direction, percent_gain]
        look_ahead (int): Number of trading intervals to look ahead for target generation
        test_size (float): test size to use for train/test split

    Methods:
        generate_target(self)
        create_train_test(self)
    """

    def __init__(
        self,
        df,
        ticker_col,
        close_col,
        features,
        target_type,
        look_ahead,
        test_size=0.2,
    ):
        self.df = df.copy()
        self.ticker_col = ticker_col
        self.close_col = close_col
        self.features = features
        self.target_type = target_type
        self.look_ahead = look_ahead
        self.test_size = test_size

    def generate_target(self):
        """
        Description:
            generates target variable for model training.
        """

        if self.target_type == "direction":
            self.df["FutureClose"] = self.df.groupby(self.ticker_col)[
                self.close_col
            ].shift(self.look_ahead)
            self.df["target"] = (
                self.df["FutureClose"] > self.df[self.close_col]
            ).astype(int)
        elif self.target_type == "percent_gain":
            self.df["FutureClose"] = self.df.groupby(self.ticker_col)[
                self.close_col
            ].shift(self.look_ahead)
            self.df["target"] = (
                self.df["FutureClose"] - self.df[self.close_col]
            ) / self.df[self.close_col]
        else:
            raise ValueError("target_type has to be one of [direction, percent_gain]")

        self.df.drop("FutureClose", axis=1, inplace=True)

    def create_train_test(self):
        self.generate_target()
        self.df = self.df[self.features + ["target"]]
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)

        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size
        )

        return X_train, X_test, y_train, y_test


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

    def __init__(self, X_train, X_test, y_train, y_test, task):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task = task

    def create_pipeline(self):

        # Identify numerical and categorical columns
        numerical_cols = self.X_train.select_dtypes(include=["number"]).columns
        categorical_cols = self.X_train.select_dtypes(exclude=["number"]).columns

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
                steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier())]
            )

        return model

    def fit(self):
        model = self.create_pipeline()
        model.fit(self.X_train, self.y_train)

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
            return_train_score=True,
            scoring="neg_mean_squared_error",
        )

        opt.fit(self.X_train, self.y_train)

        print("best params: %s" % str(opt.best_params_))
        return opt
