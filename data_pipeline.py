import pandas as pd 
import numpy as np
import yfinance as yf 
from datetime import datetime
from pandas_datareader import data as pdr

start = "2015-01-01"
end = datetime.today().strftime("%Y-%m-%d")


# getting rates
def get_fred_data():
    series = {
        "DGS2" : "2Y",
        "DGS10" : "10Y",
        "CPIAUCSL" : "CPI",
        "FEDFUNDS" : "FedFunds"
    }

    df = pd.DataFrame()

    for fred_id, name in series.items():
        df[name] = pdr.DataReader(fred_id, "fred", start, end)

    return df

# getting fx data

def get_fx_data():
    tickers = {
        "JPY=X" : "USDJPY",
        "EURUSD=X" : "EURUSD"
    }

    df = pd.DataFrame()

    for ticker, name in tickers.items():
        data = yf.download(ticker, start=start, end = end)["Close"]
        df[name]  = data

    return df

def master_data():
    rates = get_fred_data()
    fx = get_fx_data()

    df = rates.join(fx, how = "outer")
    df =  df.sort_index()
    df = df.ffill()

    return df


df = master_data()
print(df.tail())