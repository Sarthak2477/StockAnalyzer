import http.client
import json
from uagents import Agent, Context
import pandas as pd
import numpy as np
import json as json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from uagents import Agent, Context
from .analyzer_agent import Analyzer
from messages.message import Message

DataFetcher = Agent(name="data fetcher", seed="fetch data for stock prediction")
stock_symbol = None
time_series = None

# Fetch data from Alpha Vantage API
def fetch_data(stock_symbol, time_series):
    conn = http.client.HTTPSConnection("alpha-vantage.p.rapidapi.com")

    headers = {
        'X-RapidAPI-Key': "32ced199c6msh5cc64836c24756fp1e0d08jsnd83acc444637",
        'X-RapidAPI-Host': "alpha-vantage.p.rapidapi.com"
    }

    conn.request("GET", f"/query?interval=5min&function={time_series}&symbol={stock_symbol}&datatype=json&output_size=compact", headers=headers)

    res = conn.getresponse()

    if res.status == 200:  # Check if the request was successful
        data = res.read()  # Read the response content and decode it
        return data  # Return the data as a string
    else:
        return None  # Return None if the request fails


@DataFetcher.on_interval(period=30.0)
async def analyze(ctx: Context):
    global stock_symbol, time_series

    if stock_symbol is None:
        stock_symbol = input("Stock Symbol(e.g., MSFT, TSLA): ")

    if time_series is None:
        time_series = input("Time Series(e.g., TIME_SERIES_DAILY): ")
        
    data = fetch_data(stock_symbol, time_series)
    if data is not None:
        filename = 'fetched_data.json'
        json_data = data.decode("utf-8")
        with open(filename, 'w') as json_file:
            json.dump(json.loads(json_data), json_file)
        await ctx.send(Analyzer.address, Message(text="analyze"))
    else:
        print("No data fetched")
