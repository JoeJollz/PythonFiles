# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 19:17:56 2023

@author: jrjol
"""

MA_one=20
MA_two=58

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import statistics

# Define the ticker symbol
symbol = "SPY"

# Fetch the data for the S&P 500
spy_data = yf.download(symbol, start="2015-01-01", end="2023-05-31")

#Splitting the data into a test train split of 30:70
train_df, test_df = train_test_split(spy_data, test_size=0.3, shuffle=False)

def MA_ONE_today(train_df, MA_one, i):
    ma_one_today=train_df.iloc[i-MA_one+1:i+1,3].mean()
    return ma_one_today
def MA_TWO_today(train_df, MA_two, i):
    ma_two_today=train_df.iloc[i-MA_two+1:i+1,3].mean()
    return ma_two_today
def MA_ONE_yesterday(train_df, MA_one, i):
    ma_one_yesterday=train_df.iloc[i-MA_one:i,3].mean()
    return ma_one_yesterday
def MA_TWO_yesterday(train_df, MA_two, i):
    ma_two_yesterday=train_df.iloc[i-MA_two:i,3].mean()
    return ma_two_yesterday
def Sharpe_ratio_formula(year_expected_returns, Risk_free_rate):
    average_expected_return=pd.Series(year_expected_returns).mean()
    std_dev=statistics.stdev(year_expected_returns)
    Sharpe_ratio=(average_expected_return-Risk_free_rate)/std_dev
    return Sharpe_ratio

numb_trades=[]
Total_rev=[]
Success_rate=[]
max_equity_drawdown=[]
max_equity_drawup=[]
Sharpe_ratios=[]
All_stop_losses=[]
Risk_free_rate=1.85

for Stop_loss in np.arange(0.9,1.01,0.01):
    open_position=0
    revenue=0
    trade_counter=0
    Second_counter=0
    winning_trades=0
    buy_price=0
    point_increase_start_of_this_year=0
    equity_drawdowns_drawups=[]
    year_expected_returns=[]
    

    for i in range(MA_two, len(train_df)-1):
        ma_one_today=MA_ONE_today(train_df, MA_one, i)
        ma_two_today=MA_TWO_today(train_df, MA_two, i)
        ma_two_yesterday=MA_TWO_yesterday(train_df, MA_two, i)
        ma_one_yesterday=MA_ONE_yesterday(train_df, MA_one, i)
        Second_counter+=1
        
        if ma_one_today>ma_two_today and ma_one_yesterday<ma_two_yesterday and open_position==0:
            buy_price=train_df.iloc[i,3]
            open_position=1
        elif ma_one_today<ma_two_today and ma_one_yesterday>ma_two_yesterday and open_position==1:
            sell_price=train_df.iloc[i,3]
            
            profit_loss=sell_price-buy_price
            percentage_win_loss=(sell_price-buy_price)/buy_price*100
            equity_drawdowns_drawups.append(percentage_win_loss)
            
            revenue=revenue+profit_loss
            
            open_position=0
            
            trade_counter+=1
            
            if profit_loss>0:
                winning_trades+=1
            
        elif train_df.iloc[i,3]<Stop_loss*buy_price and open_position==1:
            sell_price=train_df.iloc[i,3]
            
            profit_loss=sell_price-buy_price
            percentage_win_loss=(sell_price-buy_price)/buy_price*100
            equity_drawdowns_drawups.append(percentage_win_loss)
            
            revenue=revenue+profit_loss
            
            open_position=0
            
            trade_counter+=1
            
            if profit_loss>0:
                winning_trades+=1
            
        if Second_counter==285:
            
            point_increase_this_year=revenue-point_increase_start_of_this_year
            point_increase_start_of_this_year=revenue
            mean_cost=train_df.iloc[i-285+1:i+1,3].mean()
            expected_return=point_increase_this_year/mean_cost*100
            year_expected_returns.append(expected_return)
            Second_counter=0
            
            
    if trade_counter==0:
        win_rate=0
        greatest_equity_drawdown=0
        greatest_equity_drawup=0
        Sharpe_ratio=0
        average_revenue=revenue
    else:
        win_rate=winning_trades/trade_counter
        average_revenue=revenue/trade_counter
        greatest_equity_drawdown=min(equity_drawdowns_drawups)
        greatest_equity_drawup=max(equity_drawdowns_drawups)
        Sharpe_ratio=Sharpe_ratio_formula(year_expected_returns, Risk_free_rate)
    
    All_stop_losses.append(Stop_loss)
    numb_trades.append(trade_counter)
    Success_rate.append(win_rate)
    max_equity_drawdown.append(greatest_equity_drawdown)
    max_equity_drawup.append(greatest_equity_drawup)
    Sharpe_ratios.append(Sharpe_ratio)
    Total_rev.append(revenue)
df_results=pd.DataFrame({'Stop Loss Used (as a percentage of the buy price)': All_stop_losses, 'Number of trades': numb_trades, 'Total Revenue': Total_rev, 'Success Rate': Success_rate, 'Maximum equity drawdown (Percentage)': max_equity_drawdown, 'Maximum equity drawup (Percentage)': max_equity_drawup, 'Sharpe Ratio': Sharpe_ratios })                    