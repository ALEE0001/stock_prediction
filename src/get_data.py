import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
import time
from datetime import date, datetime, timedelta
import random
import os

class GetUpdateData:
    """Get data for new tickers, or update an existing data with new dates using Twelve Data's API. This class works for free version."""
    def __init__(self, update_existing=True, existing_data_path=None, api_key=None):
        self.update_existing = update_existing
        self.existing_data_path = existing_data_path
        self.api_key = api_key
        try:
            self.existing_data = (pd.read_csv(self.existing_data_path)
                                  .sort_values(['ticker', 'datetime'], ascending=False)
                                  .drop_duplicates(subset='ticker'))
        except:
            pass

    def get_tickers(self):
        """Get S&P 500 tickers from wikipedia. Get tickers in existing data if updating instead."""
        
        df_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        df_tickers['Symbol'] = np.where(df_tickers['Symbol'].isin(['GOOG', 'GOOGL']), 'GOOGL', df_tickers['Symbol'])
        df_tickers.drop_duplicates(subset='Symbol', inplace=True)
        
        if self.update_existing:
            tickers = list(self.existing_data['ticker'])
        else:
            try:
                df_tickers = df_tickers[~df_tickers['Symbol'].isin(self.existing_data['ticker'])]
            except:
                pass
            
            limit = min(200, len(df_tickers))
            tickers = list(df_tickers.iloc[0:limit, df_tickers.columns.get_loc('Symbol')])

        return tickers, df_tickers
    
    def get_data_new(self):
        """Get new ticker data from twelve data."""
        tickers, _ = self.get_tickers()
        data_frames = []

        for ticker in tickers:
                
            try:

                url = f'https://api.twelvedata.com/earliest_timestamp?symbol={ticker}&interval=1day&apikey={self.api_key}'
                r = requests.get(url)
                
                start_date = r.text.split('"')[3]
                end_date = date.today().strftime('%Y-%m-%d')
                total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            
                batch_size = np.ceil(total_days / 5000).astype(int)
                
                l_dates = []      
                
                # If multiple batches, append start and end dates for every 5000 days
                if batch_size > 1:  
                    for i in range(batch_size-1):
                        l_dates.append((pd.to_datetime(end_date) - timedelta(days=(5000 * i))).strftime('%Y-%m-%d'))
                    l_dates.append(start_date)
                else:
                    l_dates.append(end_date)
                    l_dates.append(start_date)
                
                # Check how many credits are left, and pause for 60 seconds to reset Twelve Data's limit
                api_credit = r.headers['Api-Credits-Left']
                if api_credit == '0':
                    time.sleep(61)
                
                for i in range(batch_size):
                    try:
                        batch_end = l_dates[i]
                        batch_start = l_dates[i+1]
                        
                        # Make sure date range don't overlap throughout the loop
                        if (i > 0):
                            batch_end = (pd.to_datetime(batch_end) - timedelta(days=1)).strftime('%Y-%m-%d')
                        
                        url = f'https://api.twelvedata.com/time_series?&start_date={batch_start}&end_date={batch_end}&symbol={ticker}&format=CSV&interval=1day&apikey={self.api_key}'
                        r = requests.get(url)

                        # Decode bytes into a string
                        data_string = r.content.decode('utf-8')

                        # Use StringIO to treat the string as a file-like object
                        data_file = BytesIO(data_string.encode())
                        
                        # Union the new data to data from previous iteration. 
                        data = pd.read_csv(data_file, delimiter=';')
                        data['ticker'] = ticker
                        
                        # Append data to list
                        data_frames.append(data)
                        
                        # Check how many credits are left, and pause for 60 seconds to reset Twelve Data's limit
                        api_credit = r.headers['Api-Credits-Left']
                        if api_credit == '0':
                            time.sleep(61)
                        
                    except:
                        pass

            except:
                print(f"Couldn't download {ticker}.")
                
        return data_frames

 
    def get_data_updates(self):
        """Get updates to existing data, from twelve data."""
        tickers, _ = self.get_tickers()
        data_frames = []

        for ticker in tickers:
                
            try:
                start_date = (pd.to_datetime(self.existing_data[self.existing_data['ticker']==ticker]['datetime']) + timedelta(days=1)).dt.strftime('%Y-%m-%d')
                end_date = date.today().strftime('%Y-%m-%d')
                
                url = f'https://api.twelvedata.com/time_series?&start_date={start_date}&end_date={end_date}&symbol={ticker}&format=CSV&interval=1day&apikey={self.api_key}'
                r = requests.get(url)

                data_string = r.content.decode('utf-8')
                data_file = BytesIO(data_string.encode())
                
                data = pd.read_csv(data_file, delimiter=';')
                data['ticker'] = ticker
                
                # Append data to list
                data_frames.append(data)
                
                # Check how many credits are left, and pause for 60 seconds to reset Twelve Data's limit
                api_credit = r.headers['Api-Credits-Left']
                if api_credit == '0':
                    time.sleep(61)
                    
            except:
                print(f"Couldn't download {ticker}.")
        
        return data_frames
        
        
    def download_data(self):
        if self.update_existing:
            data_frames = self.get_data_updates()
        else:
            data_frames = self.get_data_new()
            
        df_final = pd.concat([i for i in data_frames if len(i) > 0])
        
        _, df_tickers = self.get_tickers()
        df_final = df_final.merge(df_tickers, on='ticker', how='left')

        # Only save tickers with 5 years or more data
        ticker_days = df_final.groupby('ticker')['ticker'].value_counts().reset_index()
        five_yrs_more = ticker_days[ticker_days['count'] >= 1825]['ticker']
        df_final = df_final[df_final['ticker'].isin(five_yrs_more)]

        df_final.to_csv(f'data/raw/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv', index=False)
        
GetUpdateData(update_existing=True, existing_data_path='data/data.csv', api_key=api).download_data()
