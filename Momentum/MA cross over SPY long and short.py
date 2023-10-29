# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:02:04 2023

@author: jrjol
"""

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Define the ticker symbol
symbol = "SPY"

# Fetch the data for the S&P 500
spy_data = yf.download(symbol, start="2015-01-01", end="2023-05-31")

#Splitting the data into a test train split of 30:70
train_df, test_df = train_test_split(spy_data, test_size=0.3, shuffle=False)

train_df=train_df.iloc[:,3]
test_df=test_df.iloc[:,3]

def MA_ONE_today(train_df, MA_one, i):
    ma_one_today=train_df.iloc[i-MA_one+1:i+1].mean()
    return ma_one_today
def MA_TWO_today(train_df, MA_two, i):
    ma_two_today=train_df.iloc[i-MA_two+1:i+1].mean()
    return ma_two_today
def MA_ONE_yesterday(train_df, MA_one, i):
    ma_one_yesterday=train_df.iloc[i-MA_one:i].mean()
    return ma_one_yesterday
def MA_TWO_yesterday(train_df, MA_two, i):
    ma_two_yesterday=train_df.iloc[i-MA_two:i].mean()
    return ma_two_yesterday


# first_moving_average=[]
# second_moving_average=[]
# average_rev=[]
# numb_trades=[]
# Total_rev=[]
# Success_rate=[]

MA_one=20
MA_two=50



all_trades=[]

open_short_position=0
Revenue=0
trade_counter=0

for i in range(MA_two,len(train_df)-1):
    ma_one_today=MA_ONE_today(train_df, MA_one, i)
    ma_two_today=MA_TWO_today(train_df, MA_two, i)
    ma_one_yesterday=MA_ONE_yesterday(train_df, MA_one, i)
    ma_two_yesterday=MA_TWO_yesterday(train_df, MA_two, i)
    
    if ma_two_today>ma_one_today and ma_one_yesterday>ma_two_yesterday and open_short_position==0:
        buy_short=train_df.iloc[i]
        
        open_short_position=1
    
    elif ma_two_today<ma_one_today and ma_one_yesterday<ma_two_yesterday and open_short_position==1:
        sell_short=train_df.iloc[i]
        made_from_trade=buy_short-sell_short
        Revenue=Revenue+made_from_trade
        
        all_trades.append(made_from_trade)
        trade_counter+=1
        open_short_position=0
    
    
# if trade_counter==0:
#     win_rate=0
#     average_revenue=0
# else:
#     win_rate=winning_trades/trade_counter
#     average_revenue=revenue/trade_counter
# numb_trades.append(trade_counter)
# first_moving_average.append(MA_one)
# second_moving_average.append(MA_two)
# average_rev.append(average_revenue)
# Success_rate.append(win_rate)
# Total_rev.append(revenue)

# df_results=pd.DataFrame({'First Moving Average': first_moving_average, 'Second Moving Average': second_moving_average, 'Revenue for these MAs':average_rev, 'Number of trades': numb_trades, 'Total Revenue': Total_rev, 'Success Rate': Success_rate })

            