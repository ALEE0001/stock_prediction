import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
import time
from datetime import date, datetime, timedelta
import random
import os

import json
from pandas import json_normalize


# TODO Handle update_existing=False, existing_data_path=True cases
class GetUpdateData:
    """Get data for new tickers, or update an existing data with new dates using Twelve Data's API. This class works for free version."""

    def __init__(self, update_existing=True, existing_data_path=None, api_key=None):
        self.update_existing = update_existing
        self.existing_data_path = existing_data_path
        self.api_key = api_key

        if update_existing:
            try:
                self.existing_data = (
                    pd.read_csv(self.existing_data_path, parse_dates=["datetime"])
                    .sort_values(["ticker", "datetime"], ascending=False)
                    .drop_duplicates(subset="ticker")
                )
            # except:
            #     raise ValueError("existing_data_path is either incorrect or undefined.")
            except:
                pass

    def get_tickers(self):
        """Get S&P 500 tickers from wikipedia. Get tickers in existing data if updating instead."""

        if self.update_existing:
            df_tickers = self.existing_data[
                [
                    "ticker",
                    "Security",
                    "GICS Sector",
                    "GICS Sub-Industry",
                    "Headquarters Location",
                    "Date added",
                    "CIK",
                    "Founded",
                ]
            ]
            tickers = list(self.existing_data["ticker"])
        else:

            df_tickers = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )[0]
            df_tickers = df_tickers.rename(columns={"Symbol": "ticker"})
            df_tickers["ticker"] = np.where(
                df_tickers["ticker"].isin(["GOOG", "GOOGL"]),
                "GOOGL",
                df_tickers["ticker"],
            )
            df_tickers.drop_duplicates(subset="ticker", inplace=True)

            try:
                df_tickers = df_tickers[
                    ~df_tickers["ticker"].isin(self.existing_data["ticker"])
                ]
            except:
                pass

            limit = min(200, len(df_tickers))
            tickers = list(
                df_tickers.iloc[0:limit, df_tickers.columns.get_loc("ticker")]
            )

        return tickers, df_tickers

    def get_data_new(self):
        """Get new ticker data from twelve data."""
        tickers, _ = self.get_tickers()
        data_frames = []

        for ticker in tickers:

            try:

                url = f"https://api.twelvedata.com/earliest_timestamp?symbol={ticker}&interval=1day&apikey={self.api_key}"
                r = requests.get(url)

                start_date = r.text.split('"')[3]
                end_date = date.today().strftime("%Y-%m-%d")
                total_days = (
                    pd.to_datetime(end_date) - pd.to_datetime(start_date)
                ).days

                batch_size = np.ceil(total_days / 5000).astype(int)

                l_dates = []

                # If multiple batches, append start and end dates for every 5000 days
                if batch_size > 1:
                    for i in range(batch_size - 1):
                        l_dates.append(
                            (
                                pd.to_datetime(end_date) - timedelta(days=(5000 * i))
                            ).strftime("%Y-%m-%d")
                        )
                    l_dates.append(start_date)
                else:
                    l_dates.append(end_date)
                    l_dates.append(start_date)

                # Check how many credits are left, and pause for 60 seconds to reset Twelve Data's limit
                api_credit = r.headers["Api-Credits-Left"]
                if api_credit == "0":
                    time.sleep(61)

                for i in range(batch_size):
                    try:
                        batch_end = l_dates[i]
                        batch_start = l_dates[i + 1]

                        # Make sure date range don't overlap throughout the loop
                        if i > 0:
                            batch_end = (
                                pd.to_datetime(batch_end) - timedelta(days=1)
                            ).strftime("%Y-%m-%d")

                        url = f"https://api.twelvedata.com/time_series?&start_date={batch_start}&end_date={batch_end}&symbol={ticker}&format=CSV&interval=1day&apikey={self.api_key}"
                        r = requests.get(url)

                        # Decode bytes into a string
                        data_string = r.content.decode("utf-8")

                        # Use StringIO to treat the string as a file-like object
                        data_file = BytesIO(data_string.encode())

                        # Union the new data to data from previous iteration.
                        data = pd.read_csv(data_file, delimiter=";")
                        data["ticker"] = ticker

                        # Append data to list
                        data_frames.append(data)

                        # Check how many credits are left, and pause for 60 seconds to reset Twelve Data's limit
                        api_credit = r.headers["Api-Credits-Left"]
                        if api_credit == "0":
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
                start_date = (
                    (
                        pd.to_datetime(
                            self.existing_data[self.existing_data["ticker"] == ticker][
                                "datetime"
                            ]
                        )
                        + timedelta(days=1)
                    )
                    .dt.strftime("%Y-%m-%d")
                    .values[0]
                )
                end_date = date.today().strftime("%Y-%m-%d")

                url = f"https://api.twelvedata.com/time_series?&start_date={start_date}&end_date={end_date}&symbol={ticker}&format=CSV&interval=1day&apikey={self.api_key}"
                r = requests.get(url)

                data_string = r.content.decode("utf-8")
                data_file = BytesIO(data_string.encode())

                data = pd.read_csv(data_file, delimiter=";")
                data["ticker"] = ticker

                # Append data to list
                data_frames.append(data)

                # Check how many credits are left, and pause for 60 seconds to reset Twelve Data's limit
                api_credit = r.headers["Api-Credits-Left"]
                if api_credit == "0":
                    time.sleep(61)

            except:
                print(f"Couldn't download {ticker}.")

        return data_frames

    def get_data_single_ticker(self, ticker):

        url = f"https://api.twelvedata.com/earliest_timestamp?symbol={ticker}&interval=1day&apikey={self.api_key}"
        r = requests.get(url)

        start_date = r.text.split('"')[3]
        end_date = date.today().strftime("%Y-%m-%d")

        url = f"https://api.twelvedata.com/time_series?&start_date={start_date}&end_date={end_date}&symbol={ticker}&format=CSV&interval=1day&apikey={self.api_key}"
        r = requests.get(url)

        data_string = r.content.decode("utf-8")
        data_file = BytesIO(data_string.encode())

        data = pd.read_csv(data_file, delimiter=";")
        data["ticker"] = ticker

        return data

    def download_data(self):
        if self.update_existing:
            print(f"Getting updates...")
            print(f"This may take a long time (1H) due to api limit.")
            data_frames = self.get_data_updates()
        else:
            print(f"Getting data for new tickers...")
            print(f"This may take a long time (1H) due to api limit.")
            data_frames = self.get_data_new()

        df_out = pd.concat([i for i in data_frames if len(i) > 0])

        _, df_tickers = self.get_tickers()
        df_out = df_out.merge(df_tickers, on="ticker", how="left")

        path = f'data/raw/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
        df_out.to_csv(path, index=False)
        print(f"Download completed to {path}")

    def merge_files(self):
        # List files in the directory
        directory = "data/raw/"
        files = os.listdir(directory)

        l_all_raw = []
        for i in files:
            l_all_raw.append(pd.read_csv(f"{directory}{i}"))

        df_out = pd.concat(l_all_raw).drop_duplicates(subset=["datetime", "ticker"])

        path = "data/data.csv"
        df_out.to_csv(path, index_label=False)
        print(f"Merged all raw files to {path}")


class GetDataFRED:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def get_data(self):

        data_frames = []

        dict_id = {
            "DGS10": "pc1",  # 10-Year Treasury Constant Maturity Rate (DGS10)
            "FEDFUNDS": "pc1",  # Effective Federal Funds Rate (FEDFUNDS)
            "GDP": "pc1",  # Gross Domestic Product (GDP)
            # "CPI", # Consumer Price Index (CPI)
            "UNRATE": "pc1",  # Unemployment Rate (UNRATE)
            "CP": "pc1",  # Corporate Profits After Tax (CP)
            "WM2NS": "pc1",  # Money Stock Measures  
            # "SP500": "pc1",  # S&P 500 Index (SP500)
            "NASDAQCOM": "pc1",  # NASDAQ Composite Index (NASDAQCOM)
            "UMCSENT": "pc1",  # University of Michigan Consumer Sentiment Index (UMCSENT)
            # "EXHOSLUSM495S": "lin",  # Existing Home Sales (EXHOSLUSM495S)
            "HSN1F": "pc1",  # New One Family Houses Sold: United States
            "MSPNHSUS": "pc1",  # Median Sales Price for New Houses Sold in the United States
        }

        for k, v in dict_id.items():
            print(f"Getting {k}...")
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={k}&units={v}&sort_order=asc&api_key={self.api_key}&file_type=json"
            r = requests.get(url)
            dictr = r.json()
            recs = dictr["observations"]
            temp_df = json_normalize(recs)
            temp_df["id"] = k
            temp_df = temp_df[["id", "date", "value"]]

            # Drop rows with NaN values
            temp_df["value"] = pd.to_numeric(temp_df["value"], errors="coerce")
            # temp_df = temp_df.dropna(subset=["value"])
            
            temp_df["date"] = pd.to_datetime(temp_df["date"])
            print(f"{k} [min-max] date: [{temp_df['date'].min()}-{temp_df['date'].max()}]")

            data_frames.append(temp_df)

        df_out = pd.concat([i for i in data_frames if len(i) > 0])
        
        # Create rows for missing dates (unit=day) and fill it in with previous value
        df_out["date"] = pd.to_datetime(df_out["date"])
        
        # Pivot and Forward fill missing values with the previous value
        df_out = (
            df_out.pivot_table(columns="id", values="value", index="date")
            .reset_index()
            .sort_values("date", ascending=True)
        )
        
        idx = pd.date_range(start=df_out["date"].min(), end=df_out["date"].max())
        df_out = (
            df_out.set_index("date")
            .reindex(idx)
            .rename_axis(index="datetime")
            .reset_index()
            .ffill()
        )

        return df_out

    def download_data(self):
        print(f"Getting FRED Data...")
        df_out = self.get_data()

        path = f"data/fred.csv"
        df_out.to_csv(path, index=False)
        print(f"Download completed to {path}")
