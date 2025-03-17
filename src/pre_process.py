import pandas as pd
import numpy as np
import pandas_ta as ta

class PreProcessor:
    """
    Description:
        prepares data for model training.

    Attributes:
        df (dataframe): Dataframe of the main stock data
        ticker_col (str): Column name for stock ticker/symbol
        date_col (str): Column name for date in yyyy-mm-dd format
        open_col (str): Column name for stock open price
        high_col (str): Column name for stock high price
        low_col (str): Column name for stock low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume
        look_ahead (int): Number of trading intervals to look ahead for target generation

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
        look_ahead,
        df_fred
    ):
        self.df = df
        self.ticker_col = ticker_col
        self.date_col = date_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self.look_ahead = (
            -look_ahead
        )  # Shift value. Negative looks at future, positive looks at past
        
        self.df_fred = df_fred

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
        

    def add_indicators(self):
        """Add all default indicators from ta library, then add same indicator types with custom window size (defaulted to long term: ~1 year lookback)"""

        l_df = []

        MyStrategy = ta.Strategy(
            name="MyIndicators",
            ta=[
                {"kind": "sma", "length": 10},  # Simple Moving Average
                {"kind": "ema", "length": 20},  # Exponential Moving Average
                {"kind": "rsi", "length": 14},  # Relative Strength Index
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},  # Moving Average Convergence Divergence
                {"kind": "stoch", "k": 14, "d": 3},  # Stochastic Oscillator
                {"kind": "adx", "length": 14},  # Average Directional Index
                {"kind": "cci", "length": 20},  # Commodity Channel Index
                {"kind": "atr", "length": 14},  # Average True Range
                {"kind": "bbands", "length": 20, "std": 2},  # Bollinger Bands
                {"kind": "roc", "length": 12},  # Rate of Change
                {"kind": "mfi", "length": 14},  # Money Flow Index
                {"kind": "obv"},  # On-Balance Volume
                {"kind": "vwap"},  # Volume Weighted Average Price
                {"kind": "trix", "length": 15},  # Triple Exponential Average
                {"kind": "dpo", "length": 20},  # Detrended Price Oscillator
                {"kind": "kst"},  # KST Oscillator
                {"kind": "ichimoku"},  # Ichimoku Cloud
                {"kind": "aroon", "length": 25},  # Aroon Indicator
                {"kind": "willr", "length": 14},  # Williams %R
                {"kind": "mom", "length": 10},  # Momentum
                {"kind": "psar"},  # Parabolic Stop and Reverse
                {"kind": "wma", "length": 30},  # Weighted Moving Average
                {"kind": "wma", "length": 50},  # Weighted Moving Average
                {"kind": "wma", "length": 200},  # Weighted Moving Average
                {"kind": "hma", "length": 20},  # Hull Moving Average
                {"kind": "t3", "length": 10},  # T3 Moving Average
                {"kind": "kama", "length": 10},  # Kaufman's Adaptive Moving Average
            ]
        )

        for i in self.df[self.ticker_col].unique():
            data_temp = self.df[self.df[self.ticker_col] == i]
            data_temp.set_index(self.date_col, inplace=True)
            data_temp.ta.strategy(MyStrategy, append=True, timed=True, date_col=self.date_col)

            l_df.append(data_temp)

        self.df = pd.concat(l_df)
        self.df.reset_index(inplace=True)
        
    def add_misc(self):
        """Add additional features that I think might be informative"""
        # self.df['year'] = self.df[self.date_col].dt.year
        self.df['month'] = self.df[self.date_col].dt.month
        self.df['dayofweek'] = self.df[self.date_col].dt.weekday
        
    def join_fred(self):
        """Join the FRED in to the main data"""
        self.df = self.df.merge(self.df_fred, on=self.date_col, how='left')
        
    def generate_target(self):
        """
        Description:
            generates target variable for model training.
        """

        self.df["FutureClose"] = self.df.groupby(self.ticker_col)[self.close_col].shift(
            self.look_ahead
        )

        self.df["target_direction"] = (
            self.df["FutureClose"] > self.df[self.close_col]
        ).astype(int)
        self.df["target_percent_gain"] = (
            self.df["FutureClose"] - self.df[self.close_col]
        ) / self.df[self.close_col]

        self.df.drop("FutureClose", axis=1, inplace=True)

    def pre_process(self):
        """Run all methods and save to designated path"""
        self.clean_data()
        self.add_indicators()
        self.add_misc()
        self.join_fred()
        self.generate_target()

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # self.df.dropna(inplace=True) # This is dropping cryptos because twelve data doesn't give volume info. Need to find better data source.
        # self.df = self.df[(self.df.T != 0).any()]
        self.df = self.df.sort_values([self.date_col, self.ticker_col]).reset_index(drop=True)
        self.df = self.df[self.df['GICS Sector'] != 'Cryptocurrency']
        self.df.columns = self.df.columns.str.replace('.', '_', regex=False)

        path = "data/data_model.csv"
        self.df.to_csv(path, index_label=False)
        print(f"Preprocessed data to {path}")
