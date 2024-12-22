# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:12:32 2024

@author: jrjol

This strategy aims to either trade long the best performers, and short the 
worst performers. Alternatively, this can be changed to shorting the best performers
and longing the worst performers. To consider this change, look specifically 
for the variables 'long_to_close', 'short_to_close', 'long_pairs', and 
'short_pairs', it should be quite obvious where the 'long_pair' are chosen 
based on best performing pairs, or worst performing pairs. 
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import deque

## class used later for storage of trading pairs ##

class StringStorage:
    def __init__(self, max_size=5):
        '''
        Parameters
        ----------
        max_size : INT, optional
            The number of pairs which are traded long (as well as short). 
            Hence the total number of open positions will be 10. 
            The default is 5.

        '''
        # Initialize the deque with a max size
        self.storage = deque(maxlen=max_size)

    def add_strings(self, new_strings):
        '''
        Parameters
        ----------
        new_strings : LIST OR TUPLE
            This will contain the FX pairs which are to be opened or remain open.

        Returns
        -------
        missing_strings : SET
            Returning the FX pairs whose position must now be closed.

        '''
        # First, check which string is missing
        current_strings = set(self.storage)  # Convert current storage to a set
        new_strings_set = set(new_strings)   # Convert new strings to a set
        
        # Identify the missing string(s) from the old set
        missing_strings = current_strings - new_strings_set
        
        # If there's a string missing, replace it
        for missing in missing_strings:
            self.storage.remove(missing)  # Remove the missing string
        
        # Add the new strings (make sure to only add what's not already in the storage)
        for new_string in new_strings:
            if new_string not in self.storage:
                self.storage.append(new_string)
            
        return missing_strings

    def get_storage(self):
        # Return the current storage
        return list(self.storage)
    
## class coding finished ##



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

cumulative_returns_df = pd.DataFrame()

# Calculate and plot cumulative returns for each currency pair
plt.figure(figsize=(12, 8))

for pair in formatted_pairs:
    if pair in fx_data:
        data = fx_data[pair]
        # Calculate daily returns
        data['Daily Return'] = data['Close'].pct_change()
        # Calculate cumulative returns
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
        # Store cumulative returns in the DataFrame
        cumulative_returns_df[pair.replace('=X', '')] = data['Cumulative Return']
        # Plot cumulative returns
        plt.plot(data.index, data['Cumulative Return'], label=pair.replace('=X', ''))
        plt.xlabel('Date')
        plt.ylabel('Return')
    else:
        print(f"No data for {pair}")
        

'''
Now the FX data is imported , build a basic trading strategy.
Test a variety of lookback periods, and holding periods. 

'''

lookback_periods = [3, 5, 10, 15 ]
holding_periods = [2,3, 5, 10, 15]



for lookback in lookback_periods:
    for holding_period in holding_periods:
        
        long_pairs = StringStorage()
        short_pairs = StringStorage()
        
        for i in np.arange(lookback, len(data)/100, holding_period):
            pct_changes = {}
            i = int(i)
            for pair, data in fx_data.items():    
                data = data['Close']
                pct_change = data.iloc[i]/ data.iloc[i-lookback]
                
                pct_changes[pair] = pct_change
                
            sorted_pct_changes = sorted(pct_changes.items(), key=itemgetter(1))
            worst_perf = sorted_pct_changes[:5]
            best_perf = sorted_pct_changes[(len(sorted_pct_changes)-5):]
            
            long_to_close = long_pairs.add_strings(best_perf)
            short_to_close = short_pairs.add_strings(worst_perf)
            
            
                
            
            
            

    
    
    