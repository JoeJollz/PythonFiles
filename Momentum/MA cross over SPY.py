# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 20:26:23 2023

@author: jrjol
"""

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import statistics

symbol="SPY"

spy_data=yf.download(symbol, start="2015-01-01", end="2023-05-31")

train_df, test_df=train_test_split(spy_data, test_size=0.3, shuffle=False)

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
def Sharpe_ratio_formula(year_expected_returns, Risk_free_rate):
    average_expected_return=pd.Series(year_expected_returns).mean()
    std_dev=statistics.stdev(year_expected_returns)
    Sharpe_ratio=(average_expected_return-Risk_free_rate)/std_dev
    return Sharpe_ratio
    


first_moving_average=[]
second_moving_average=[]
average_point=[]
numb_trades=[]
Total_point=[]
Success_rate=[]
max_equity_drawdown=[]
max_equity_drawup=[]
Sharpe_ratios=[]
Risk_free_rate=1.85

for MA_one in range(5,49):
    
    for MA_two in range(15,120):
        
        open_position=0
        trade_counter=0
        points=0
        point_increase_start_of_this_year=0
        point_increase_this_year=0
        winning_trades=0
        second_counter=0
        equity_drawdowns_drawups=[]
        year_expected_returns=[]
        

        for i in range(MA_two,len(train_df)-1):
            ma_one_today=MA_ONE_today(train_df, MA_one, i)
            ma_two_today=MA_TWO_today(train_df, MA_two, i)
            ma_one_yesterday=MA_ONE_yesterday(train_df, MA_one, i)
            ma_two_yesterday=MA_TWO_yesterday(train_df, MA_two, i)
            second_counter+=1
            
            
            if ma_one_today>ma_two_today and ma_one_yesterday<ma_two_yesterday and open_position==0:
                buy_price=train_df.iloc[i]
                
                open_position=1
            elif ma_one_today<ma_two_today and ma_one_yesterday>ma_two_yesterday and open_position==1:
                sell_price=train_df.iloc[i]
                
                profit_loss=sell_price-buy_price
                percentage_win_loss=(sell_price-buy_price)/buy_price*100
                equity_drawdowns_drawups.append(percentage_win_loss)
                
                points=points+profit_loss
                
                open_position=0
                
                trade_counter+=1
                
                if profit_loss>0:
                    winning_trades+=1
                    
            if second_counter==252:
                
                point_increase_this_year=points-point_increase_start_of_this_year
                point_increase_start_of_this_year=points
                mean_cost=train_df.iloc[i-252+1:i+1].mean()
                expected_return=point_increase_this_year/mean_cost*100
                year_expected_returns.append(expected_return)
                second_counter=0
                
        
                
                        
        if trade_counter==0:
            win_rate=0
            greatest_equity_drawdown=0
            greatest_equity_drawup=0
            Sharpe_ratio=0
            average_point=points
        else:
            win_rate=winning_trades/trade_counter
            average_revenue=points/trade_counter
            greatest_equity_drawdown=min(equity_drawdowns_drawups)
            greatest_equity_drawup=max(equity_drawdowns_drawups)
            Sharpe_ratio=Sharpe_ratio_formula(year_expected_returns, Risk_free_rate)
            
            
        numb_trades.append(trade_counter)
        first_moving_average.append(MA_one)
        second_moving_average.append(MA_two)
        average_point.append(average_revenue)
        Success_rate.append(win_rate)
        max_equity_drawdown.append(greatest_equity_drawdown)
        max_equity_drawup.append(greatest_equity_drawup)
        Sharpe_ratios.append(Sharpe_ratio)
        Total_point.append(points)
            
df_results=pd.DataFrame({'First Moving Average': first_moving_average, 'Second Moving Average': second_moving_average, 'Revenue for these MAs':average_point, 'Number of trades': numb_trades, 'Total Revenue': Total_point, 'Success Rate': Success_rate, 'Maximum equity drawdown (Percentage)': max_equity_drawdown, 'Maximum equity drawup (Percentage)': max_equity_drawup, 'Sharpe Ratio': Sharpe_ratios })            