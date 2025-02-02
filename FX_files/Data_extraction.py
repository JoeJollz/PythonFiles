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

'''
Currently re working the indexing for the backtesting, DateTime based, rather than
simple loop. Need to reconsider approach.
Consider 
- how to extract a datetime from our current data set, which we can loop over.
- raise expections for missing data at specific dates (e.g. weekends).
Messy code, needs cleaning up with additional comments added. 
'''

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
        This method is crucial and key for identifying any new pairs which must
        now have a position open, or any old pairs which must now be closed.
        
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
        strings_to_close = current_strings - new_strings_set # correct
        strings_to_open = new_strings_set - current_strings # correct
        # print('----------------')
        # print('new_strings', new_strings_set)
        # print('old_strings', current_strings)
        # print('the missing strings', missing_strings)
        # print('----------------')
        # print('new_strings', new_strings_set)
        # print('old_strings', current_strings)
        # print('the missing strings', strings_to_open)
        
        
        # If there's a string missing, replace it
        for missing in strings_to_close:
            self.storage.remove(missing)  # Remove the missing string
        
        # Add the new strings (make sure to only add what's not already in the storage)
        for new_string in new_strings:
            if new_string not in self.storage:
                self.storage.append(new_string)
            
        return strings_to_close, strings_to_open

    def get_storage(self):
        # Return the current storage
        return list(self.storage)
    
    
## this class is reponsible for the order book, and maintains what positions 
# are currently open.
class PositionBook:
    def __init__(self):
        self.positions = {} # simply create dictionary
        
        
    def organise_positions(self, to_open, to_close, index, fx_data , long=True):
        '''
        This method is updates our position book. It will consider if the book
        is for long positions of short position. It will consider what
        positions will be closed and therefore caculate their relavent profit
        or loss. It will then remove the positions from our order book, and 
        then added the new positions which are to opened, with their price action.
        It will store the FX_pair name as the key, with the price action at the
        point of opening the position being the value. 
        
        It will then return the average win / loss for all the positions opened
        and closed. 
        
        This method must be called seperately the long positions order book, 
        and the short positions order book.   ??? 

        Parameters
        ----------
        to_open : SET
            All the new positions which are to be opened.
        to_close : SET
            All the old positions which must be closed.
        index : INT
            Index representing the current time in the historical data.
        fx_data : DICT
            Contains all the FX pair trading candle data.
        long : Boolean, optional
            This will determine if the book is for the long positions or short 
            positions. The default is True.

        Returns
        -------
        average_win_loss : FLOAT
            This is the average win/loss for all the positions closed 
            for the current time step. NOTE THE 'AVERAGE' IS NOT TRUE, FURTHER 
            WORK IS TO BE COMPLETED IN ORDER TO ACCURATELY UPDATE OUR NEW 
            TRADING CAPITAL VALUE - THIS WILL BE DEPENDENT ON THE WEIGHTING OF
            OUR POSITIONS AND THE NUMBER OF POSITIONS CLOSED DURING THIS TIME
            PERIOD. 

        '''
        
        book = self.positions
        
        trades = []
        w_trades = 0
        l_trades = 0
        
        for _ in to_close:
            open_price = book[_]
            close_price = fx_data[_]['Close'].loc[index]
            del book[_] # removing the FX pair from our position book
            
            if long == True:
                pct_gain_loss = close_price/open_price 
            else:
                pct_gain_loss = open_price/close_price
            
            trades.append(pct_gain_loss)
            
            if pct_gain_loss>0:
                w_trades+=1
            elif pct_gain_loss<0:
                l_trades+=1
            
            
        trade_counter = len(to_close)
        
        for _ in to_open:
            book[_] = fx_data[_]['Close'].loc[index] # storing the new price of the newly opened poistion
            
    
        return trades, trade_counter, w_trades, l_trades
    
## this class is responsible for storing the trading metrics of the strategy

class results_counter:
    def __init__(self):
        self.a = 0 # TBF, need to make a metric counter for various metrics.... !!!! 

   
class results_store:
    def __init__(self):
        self.results = [] # open simple list to store result information
    
    def store_result(self, addition):
        
        for _ in addition:
            self.results.append(_)
        
        #self.results.append(addition)  # this can be used to store a tuple of input parameters, or metric input
    
    def organise_trade_results(self):
        
        # mini = min(self.results) # minimal value of the metrics
        # maxi = max(self.results) # maximum value of metrics
        
        # mini_index = self.results.index(mini) # index of the smallest metric
        # maxi_index = self.results.index(maxi) # index of the largest metric
        
       # mini_parameter = parameter_store[mini_index] # strategy parameters corresponding to the small metric
       # maxi_parameter = parameter_store[maxi_index] # strategy parameters corresponding to the largest metri
       
       
        w_trades = 0
        l_trades = 0
        eq = 1
       
        for _ in self.results:
            eq *= _
            
            if _ >1:
                w_trades +=1
            elif _<1:
                l_trades +=1
        
        n_trades = w_trades + l_trades
        
        win_rate = w_trades/n_trades*100
        
        avg_rt = (eq-1)/n_trades
            
        return eq, avg_rt, w_trades, l_trades, n_trades, win_rate
    

    
    
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

#date_range = pd.date_range(start_date,end_date)

# fetching the data from YF #
def fetch_data(pairs, start_date, end_date):
    
    forex_data = {} # dict to store the FX candles for each pair
    
    for pair in pairs:
        ticker = yf.Ticker(pair)
        data = ticker.history(start=start_date, end=end_date,period ="1d")
        
        if not data.empty:
            forex_data[pair] = data
        
        #forex_data[pair].index = forex_data[pair].index.normalize()
    
        data.index = data.index.tz_localize(None)    
    
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

# strategy_parameters = results()
# total_returns = results()

min_rows = min(df.shape[0] for df in fx_data.values()) # identify the maximum size of loop



for lookback in lookback_periods:
    date_range = fx_data["AUDCAD=X"].index[lookback:]
    
    for holding_period in holding_periods:
        
        long_pairs = StringStorage()
        short_pairs = StringStorage()
        
        long_book = PositionBook()
        short_book = PositionBook()
        trades_book_l = results_store()
        trades_book_s = results_store()
        
        #date_range = pd.date_range(start=start_date, end=end_date, freq=f'{holding_period}D')
        
        for current_date in date_range: # need to consider how holding period will be taken in to account.
            lookback_date = current_date - pd.Timedelta(days=lookback)
            #lookback_date = lookback_date.normalize()
            #current_date = current_date.normalize()
        #for i in np.arange(lookback, min_rows, holding_period):
            pct_changes = {}
            #i = int(i)
            for pair, data in fx_data.items(): 
                
                if current_date and lookback_date in data['Close'].index:
                    print('problem pair', pair, ' Date issues could be: ', current_date, lookback_date)
                    close_price = data['Close'].loc[current_date]
                
                else:
                    print(f"Date {current_date} not found in the index for {pair}.")
                    close_price = None  # Handle as appropriate
                
                if current_date in data.index and lookback_date in data.index:
                    print('success', pair)
                    #data.index = data.index.normalize()
                    data = data['Close']
                    
                    pct_change = data.loc[current_date] / data.loc[lookback_date]
                    #pct_change = data.iloc[i]/ data.iloc[i-lookback] # increase/decrease
                    
                    pct_changes[pair] = pct_change
                else:
                    print(f"Missing data for {pair} on {current_date} or {lookback_date}")
                
            sorted_pct_changes = sorted(pct_changes.items(), key=itemgetter(1)) # sort the pairs, from worst (top) to best (bottom)
            
            worst_perf = sorted_pct_changes[:5] # worst 5 performers.
            worst_perf = [_[0] for _ in worst_perf] # extracting the first element of the tuples in the list, which corresponds to the FX pair name.
            best_perf = sorted_pct_changes[(len(sorted_pct_changes)-5):] # top 5 performers
            best_perf = [_[0] for _ in best_perf]
            
            
            long_to_close, long_to_open = long_pairs.add_strings(best_perf)
            short_to_close, short_to_open = short_pairs.add_strings(worst_perf)
            
            long_trades, counter_l, w_c_l, l_c_l = long_book.organise_positions(long_to_open, 
                                                                long_to_close, 
                                                                current_date, 
                                                                fx_data, 
                                                                long=True) # call class method
            
            short_trades, counter_s, w_c_s, l_c_s = short_book.organise_positions(short_to_open, 
                                                                short_to_close, 
                                                                current_date, 
                                                                fx_data, 
                                                                long=False)
            
            trades_book_l.store_result(long_trades)
            trades_book_s.store_result(short_trades)
            
            # if len(long_to_close) != 0: # ensuring trades are occuring, before storing
                
            #     trades_book.append(averge_win_loss_0/len(long_to_close))
        
trades_long = trades_book_l.results
trades_short = trades_book_s.results

eq_l, avg_rt_l, w_trades_l, l_trades_l, n_trades_l, win_rate_l = trades_book_l.organise_trade_results()

eq_s, avg_rt_s, w_trades_s, l_trades_s, n_trades_s, win_rate_s = trades_book_s.organise_trade_results()


'''
continuation notes,
many issues have been debugged. 

Need to formalise a metric store for a trading strategy parameter combination.
Hence, we can then identify the optimal trading metrics.

Proof checking of the strategy and classes needs to be performed. specifically the
correct closing and opening of positions. 
'''     