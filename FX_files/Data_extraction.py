# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:12:32 2024

@author: jrjol
"""

import pandas as pd
import numpy as np
import yfinance as yf

currency_pairs = [
    "EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD", "EUR/JPY",
    "GBP/JPY", "GBP/AUD", "GBP/CAD", "GBP/NZD", "GBP/CHF",
    "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
    "NZD/JPY", "NZD/CAD", "NZD/CHF",
    "CAD/JPY", "CAD/CHF",
    "CHF/JPY",
    "USD/KRW", "EUR/KRW", "GBP/KRW", "AUD/KRW", "CAD/KRW", 
    "CHF/KRW", "NZD/KRW", "JPY/KRW",
    "USD/HKD", "USD/ZAR", "USD/THB",
    "USD/MXN", "USD/DKK", "USD/NOK", "USD/SEK", "USD/PLN", "USD/CZK",
    "EUR/ZAR", "EUR/NOK", "EUR/SEK", "EUR/DKK", "EUR/HUF", "EUR/PLN",
    "GBP/ZAR",
    "AUD/ZAR",
    "CHF/ZAR"
]


# format the FX for YF to read #
def format_pairs(pair_list):
    
    return [pair.replace('/', '') + '=X' for pair in pair_list]

formatted_pairs = format_pairs(currency_pairs)

# date range #
start_date = '2010-01-01'
end_date = '2020-12-31'

# fetching the data from YF #
def fetch_data(pairs, start_date, end_date):
    
    forex_data = {} # dict to store the FX candles for each pair
    
    for pair in pairs:
        ticker = yf.Ticker(pair)
        data = ticker.history(start=start_date, end=end_date,period ="1d")
        
        if not data.empty:
            forex_data[pair] = data
    
    return forex_data

fx_data = fetch_data(formatted_pairs, start_date, end_date)
    
    
    