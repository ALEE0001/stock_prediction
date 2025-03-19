import pandas as pd
import numpy as np
import pandas_ta as ta

class PreProcessor:
    def __init__(self, df, ticker_col, date_col, open_col, high_col, low_col, close_col, volume_col, look_ahead, df_fred):
        self.df = df.copy()
        self.ticker_col = ticker_col
        self.date_col = date_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.look_ahead = -look_ahead
        self.df_fred = df_fred

    def clean_data(self):
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df.sort_values(by=[self.ticker_col, self.date_col], inplace=True)
        self.df = self.df[self.df[self.date_col] >= pd.to_datetime('1990-01-01')]

        ticker_days = self.df.groupby(self.ticker_col).size().reset_index(name='count')
        five_yrs_more = ticker_days[ticker_days["count"] >= 1825][self.ticker_col]
        self.df = self.df[self.df[self.ticker_col].isin(five_yrs_more)]
        self.df['GICS Sector'] = self.df['GICS Sector'].fillna('Cryptocurrency')
        self.df = self.df[self.df['GICS Sector'] != 'Cryptocurrency'] # Remove cryptos for now, TODO Find better datasource

        self.df["PrevClose"] = self.df.groupby(self.ticker_col)[self.close_col].shift(1)
        self.df['1DPercChange'] = (self.df[self.close_col] - self.df['PrevClose']) / self.df['PrevClose']
        weird_tickers = self.df[self.df['1DPercChange'] >= 1][self.ticker_col].unique()
        cryptos = self.df[self.df['GICS Sector'] == 'Cryptocurrency'][self.ticker_col].unique()
        weird_tickers = [i for i in weird_tickers if i not in cryptos]
        self.df = self.df[~self.df[self.ticker_col].isin(weird_tickers)]
        self.df.drop(['PrevClose', '1DPercChange'], axis=1, inplace=True)

    def add_indicators(self):
        MyStrategy = ta.Strategy(
            name="MyIndicators",
            ta=[
                {"kind": "sma", "length": 10},
                {"kind": "ema", "length": 20},
                {"kind": "rsi", "length": 14},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "stoch", "k": 14, "d": 3},
                {"kind": "adx", "length": 14},
                {"kind": "cci", "length": 20},
                {"kind": "atr", "length": 14},
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "roc", "length": 12},
                {"kind": "mfi", "length": 14},
                {"kind": "obv"},
                {"kind": "vwap"},
                {"kind": "trix", "length": 15},
                {"kind": "dpo", "length": 20},
                {"kind": "kst"},
                {"kind": "ichimoku"},
                {"kind": "aroon", "length": 25},
                {"kind": "willr", "length": 14},
                {"kind": "mom", "length": 10},
                {"kind": "psar"},
                {"kind": "wma", "length": 30},
                {"kind": "wma", "length": 50},
                {"kind": "wma", "length": 200},
                {"kind": "hma", "length": 20},
                {"kind": "t3", "length": 10},
                {"kind": "kama", "length": 10},
            ]
        )

        self.df.set_index(self.date_col, inplace=True)
        self.df.ta.strategy(MyStrategy, append=True, timed=True)
        self.df.reset_index(inplace=True)

    def add_misc(self):
        self.df['month'] = self.df[self.date_col].dt.month
        self.df['dayofweek'] = self.df[self.date_col].dt.weekday

    def join_fred(self):
        self.df = self.df.merge(self.df_fred, on=self.date_col, how='left')

    def generate_target(self):
        self.df["FutureClose"] = self.df.groupby(self.ticker_col)[self.close_col].shift(self.look_ahead)
        self.df["target_direction"] = np.where(self.df["FutureClose"] > self.df[self.close_col], 1, 0)
        self.df["target_percent_gain"] = (self.df["FutureClose"] - self.df[self.close_col]) / self.df[self.close_col]
        self.df.drop("FutureClose", axis=1, inplace=True)

    def handle_missing(self):
        self.df.sort_values(by=[self.ticker_col, self.date_col], inplace=True)
        columns_to_ffill = self.df.select_dtypes(include=[np.number]).columns.difference(['target_direction', 'target_percent_gain'])
        self.df[columns_to_ffill] = self.df.groupby(self.ticker_col)[columns_to_ffill].ffill()
        self.df.reset_index(drop=True, inplace=True)
        self.df.dropna(inplace=True)

    def pre_process(self):
        self.clean_data()
        self.add_indicators()
        self.add_misc()
        self.join_fred()
        self.generate_target()
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.handle_missing()
        self.df.columns = self.df.columns.str.replace('.', '_', regex=False)
        path = "data/data_model.csv"
        self.df.to_csv(path, index=False)
        print(f"Preprocessed data saved to {path}")
