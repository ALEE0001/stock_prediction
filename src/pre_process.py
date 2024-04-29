import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna

class PreProcessor:
    def __init__(self, df, window_slow=252, window_fast=126, window_sign=21):
        self.df = df
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign
        
    def clean_data(self):
        """Prep raw data for further processing"""
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        self.df = dropna(self.df)
        
        # Only include tickers with 5 years or more observations
        ticker_days = self.df.groupby('ticker')['ticker'].value_counts().reset_index()
        five_yrs_more = ticker_days[ticker_days['count'] >= 1825]['ticker']
        self.df = self.df[self.df['ticker'].isin(five_yrs_more)]
        
        # Get founded year from messy Founded column, using regex
        self.df['Founded'] = self.df['Founded'].astype(str)        
        ticker_founded = (self.df.groupby('ticker')['Founded']
                        .apply(lambda x: x.str.extractall(r'(\d{4,})').astype(int))
                        .reset_index()
                        .groupby('ticker')
                        .agg({0:'min'})
                        .rename(columns={0:'founded_regex'})
                        .reset_index())

        self.df = self.df.merge(ticker_founded, on='ticker', how='left')

        # Remove records that are before the founded date
        self.df = self.df[self.df['datetime'].dt.year >= self.df['founded_regex']]
        self.df.drop(['Founded', 'founded_regex'], axis=1, inplace=True)
        
        self.df.sort_values(['ticker', 'datetime'], ascending=True, inplace=True)
        
    def add_indicators(self):
        """ Add all default indicators from ta library, then add same indicator types with custom window size (defaulted to long term: ~1 year lookback)"""
        self.open = self.df['open']
        self.high = self.df['high']
        self.low = self.df['low']
        self.close = self.df['close']
        self.volume = self.df['volume']
        
        l_df = []
        
        for i in self.df['ticker'].unique():
            data_temp = add_all_ta_features(self.df[self.df['ticker']==i], open="open", high="high", low="low", close="close", volume="volume", fillna=True)
            
            # data_temp['momentum_ao_cust'] = ta.momentum.AwesomeOscillatorIndicator(high=self.high, low=self.low, window1=self.window_sign, window2=self.window_fast, fillna=False)
            # data_temp['momentum_kama_cust'] = ta.momentum.KAMAIndicator(close=self.close, window=self.window_fast, pow1=self.window_sign, pow2=self.window_slow, fillna=False)
            # data_temp['momentum_ppo_cust'] = ta.momentum.PercentagePriceOscillator(close=self.close, window_slow=self.window_slow, window_fast=self.window_fast, window_sign=self.window_sign, fillna=False)
            # data_temp['momentum_pvo_cust'] = ta.momentum.PercentageVolumeOscillator(volume=self.volume, window_slow=self.window_slow, window_fast=self.window_fast, window_sign=self.window_sign, fillna=False)
            # data_temp['momentum_roc_cust'] = ta.momentum.ROCIndicator(close=self.close, window=self.window_fast, fillna=False)
            # data_temp['momentum_rsi_cust'] = ta.momentum.RSIIndicator(close=self.close, window=self.window_fast, fillna=False)
            # data_temp['momentum_stoch_rsi_cust'] = ta.momentum.StochRSIIndicator(close=self.close, window=self.window_fast, smooth1=self.window_sign, smooth2=self.window_sign, fillna=False)
            # data_temp['momentum_stoch_o_cust'] = ta.momentum.StochasticOscillator(high=self.high, low=self.low, close=self.close, window=self.window_fast, smooth_window=self.window_sign, fillna=False)
            
            l_df.append(data_temp)
        
        df_out = pd.concat(l_df)
        
        return df_out
                
        
    def pre_process(self):
        """Clean and Add Indicators"""
        self.clean_data()
        df_out = self.add_indicators()
        df_out.dropna(inplace=True)
        
        path = 'data/data_model.csv'
        df_out.to_csv(path, index_label=False)
        print(f'Preprocessed data to {path}')