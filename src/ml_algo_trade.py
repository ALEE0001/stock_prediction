import pandas as pd
import numpy as np
from timedelta import Timedelta 

from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset
from datetime import datetime 
from alpaca_trade_api import REST 

from src.get_data import GetUpdateData
from src.get_data import GetDataFRED
from ta import add_all_ta_features

from joblib import dump, load
import json

# Load secrets from secrets.json
with open('secrets.json') as f:
    secrets = json.load(f)
    
api_key_twelve = secrets['default']['twelve_data_api']
api_key_fred = secrets['default']['fred_api']

api_key_alpaca = secrets['default']['alpaca_api_key']
api_secret_alpaca = secrets['default']['alpaca_api_secret']

clf_model = load("models/candidate_models/lgbm-direction-classifier-2024-05-13-19-23-24.joblib")

class MLStrategy(Strategy):
    def initialize(self, clf_model=None, symbol: str = "SPY", cash_at_risk: float = 1):
        self.symbol = symbol
        self.sleeptime = "7D"
        self.last_trade = None
        # self.set_market("24/7")
        self.cash_at_risk = cash_at_risk

    def get_data(self):

        ticker = self.symbol.replace("-", "/")
        df_stock = (
            GetUpdateData(api_key=api_key_twelve)
            .get_data_single_ticker(ticker)
            .sort_values("datetime", ascending=True)
        )

        _, df_tickers = GetUpdateData(
            update_existing=False, api_key=api_key_twelve
        ).get_tickers()

        df_stock = df_stock.merge(
            df_tickers[["ticker", "GICS Sector", "GICS Sub-Industry"]],
            on="ticker",
            how="left",
        ).drop('ticker', axis=1)
        
        df_fred = GetDataFRED(api_key=api_key_fred).get_data()
        df_fred["datetime"] = pd.to_datetime(df_fred["datetime"])

        df_ohlci = add_all_ta_features(
            df_stock,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True,
        )

        df_ohlci["datetime"] = pd.to_datetime(df_ohlci["datetime"])
        df_ohlci["month"] = df_ohlci["datetime"].dt.month
        df_ohlci["dayofweek"] = df_ohlci["datetime"].dt.weekday
        df_ohlci = df_ohlci.merge(df_fred, on="datetime", how="left")



        print(df_ohlci["datetime"].min())

        return df_ohlci

    def get_date(self):
        today = self.get_datetime()
        return today.strftime("%Y-%m-%d")

    def get_signal(self):
        X = self.df_ohlci.copy()
        # today = self.get_date()
        # X = X.reset_index(drop=True)
        # prev_index = X[X["datetime"] == today].index[0] - 1
        # X = X.iloc[[prev_index]]
        X = X.tail(1)
        X.drop("datetime", axis=1, inplace=True)

        pred_direction_proba = clf_model.predict_proba(X)[0][1]

        return pred_direction_proba

    def position_sizing(self):
        cash = self.cash
        last_price = self.get_last_price(self.symbol)
        quantity = cash * self.cash_at_risk // last_price
        return cash, last_price, quantity

    def on_trading_iteration(self):
        if broker.is_market_open():
            self.df_ohlci = self.get_data()
            _, last_price, _ = self.position_sizing()
            print(f"Last Price: {last_price:.2f}")
            
            pred_direction_proba = self.get_signal()
            yesterday_close_price = X["close"].values[0]
            print(f"pred_direction_proba: {pred_direction_proba}")

            if (pred_direction_proba >= 0.5):
                if self.last_trade == "sell":
                    self.sell_all()
                cash, last_price, quantity = self.position_sizing()
                
                if cash > last_price:
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=last_price * 1.10,
                        stop_loss_price=last_price * 0.95,
                    )
                    self.submit_order(order)
                    self.last_trade = "buy"

            if (pred_direction_proba < 0.5): 
                if self.last_trade == "buy":
                    self.sell_all()
                cash, last_price, quantity = self.position_sizing()
                if cash > last_price:
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "sell",
                        type="bracket",
                        take_profit_price=last_price * 0.90,
                        stop_loss_price=last_price * 1.05,
                    )
                    self.submit_order(order)
                    self.last_trade = "sell"
        else:
            print("market is closed.")