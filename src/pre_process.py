import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna


class PreProcessor:
    """
    Description:
        prepares data for model training.

    Attributes:
        df (dataframe):
        ticker_col (str): Column name for stock ticker/symbol
        date_col (str): Column name for date in yyyy-mm-dd format
        open_col (str): Column name for stock open price
        high_col (str): Column name for stock high price
        low_col (str): Column name for stock low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume

    Methods:
        generate_target(self)
        create_train_test(self)
    """
    def __init__(
        self,
        df,
        ticker_col,
        date_col,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col,
    ):
        self.df = df
        self.ticker_col = ticker_col
        self.date_col = date_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

    def clean_data(self):
        """Prep raw data for further processing"""
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        # Remove records before 1990
        self.df = self.df[self.df[self.date_col] >= pd.to_datetime('1990-01-01')]

        # Only include tickers with 5 years or more observations
        ticker_days = (
            self.df.groupby(self.ticker_col)[self.ticker_col]
            .value_counts()
            .reset_index()
        )
        five_yrs_more = ticker_days[ticker_days["count"] >= 1825][self.ticker_col]
        self.df = self.df[self.df[self.ticker_col].isin(five_yrs_more)]
        
        self.df['GICS Sector'] = self.df['GICS Sector'].fillna('Cryptocurrency')

        # Identify uncaught stock splits/weirdness (Daily gain of 100% or more) and remove these tickers.
        self.df["PrevClose"] = self.df.groupby(self.ticker_col)[self.close_col].shift(1)
        self.df['1DPercChange'] = (self.df[self.close_col] - self.df['PrevClose']) / self.df['PrevClose']
        weird_tickers =  self.df[self.df['1DPercChange'] >= 1][self.ticker_col].unique()
        
        cryptos = self.df[self.df['GICS Sector'] == 'Cryptocurrency']['ticker'].unique()
        weird_tickers = [i for i in weird_tickers if i not in cryptos] # Exclude cryptos from this
        self.df = self.df[~self.df[self.ticker_col].isin(weird_tickers)]
        self.df.drop(['PrevClose', '1DPercChange'], axis=1, inplace=True)

        self.df.sort_values(
            [self.ticker_col, self.date_col], ascending=True, inplace=True
        )
        

    def add_indicators(self, window_slow=252, window_fast=126, window_sign=21):
        """Add all default indicators from ta library, then add same indicator types with custom window size (defaulted to long term: ~1 year lookback)"""
        base_cols = [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col]

        l_df = []

        for i in self.df[self.ticker_col].unique():
            data_temp = add_all_ta_features(
                self.df[self.df[self.ticker_col] == i],
                open=self.open_col,
                high=self.high_col,
                low=self.low_col,
                close=self.close_col,
                volume=self.volume_col,
                fillna=True,
            )

            # data_temp['momentum_ao_cust'] = ta.momentum.AwesomeOscillatorIndicator(high=self.high, low=self.low, window1=self.window_sign, window2=self.window_fast, fillna=False)
            # data_temp['momentum_kama_cust'] = ta.momentum.KAMAIndicator(close=self.close, window=self.window_fast, pow1=self.window_sign, pow2=self.window_slow, fillna=False)
            # data_temp['momentum_ppo_cust'] = ta.momentum.PercentagePriceOscillator(close=self.close, window_slow=self.window_slow, window_fast=self.window_fast, window_sign=self.window_sign, fillna=False)
            # data_temp['momentum_pvo_cust'] = ta.momentum.PercentageVolumeOscillator(volume=self.volume, window_slow=self.window_slow, window_fast=self.window_fast, window_sign=self.window_sign, fillna=False)
            # data_temp['momentum_roc_cust'] = ta.momentum.ROCIndicator(close=self.close, window=self.window_fast, fillna=False)
            # data_temp['momentum_rsi_cust'] = ta.momentum.RSIIndicator(close=self.close, window=self.window_fast, fillna=False)
            # data_temp['momentum_stoch_rsi_cust'] = ta.momentum.StochRSIIndicator(close=self.close, window=self.window_fast, smooth1=self.window_sign, smooth2=self.window_sign, fillna=False)
            # data_temp['momentum_stoch_o_cust'] = ta.momentum.StochasticOscillator(high=self.high, low=self.low, close=self.close, window=self.window_fast, smooth_window=self.window_sign, fillna=False)

            l_df.append(data_temp)

        self.df = pd.concat(l_df)

    def pre_process(self):
        """Run all methods and save to designated path"""
        self.clean_data()
        self.add_indicators()
        self.df.dropna() # This is dropping cryptos because twelve data doesn't give volume info. Need to find better data source.and
        self.df = self.df[(self.df.T != 0).any()]
        

        path = "data/data_model.csv"
        self.df.to_csv(path, index_label=False)
        print(f"Preprocessed data to {path}")
