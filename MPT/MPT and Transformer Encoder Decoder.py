# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:58:25 2023

@author: jrjol
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="max_sharpe transforms the optimization problem")
from cvxpy import SCS

### Importing 20 ticker data. This includes close, open ,high etc
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
           'JPM', 'BAC', 'WMT', 'JNJ', 'V', 'PG', 
           'XOM', 'VZ', 'UNH', 'T', 'HD', 'DIS', 
           'PFE', 'TSLA', 'MA']

df = yf.download(tickers, start="2019-11-22", end="2023-05-31")
df = df.dropna(axis=1)


#### extracting just the adj close. We need to predict this value if you can,
# Using the mulitvariate dataset in df. We also need this true values, so we 
# can make the returns prediction.
# (Adj_close_pred_t+1-Adj_close_today_t)/Adj_close_today_t
adj_close_df=df['Adj Close']


## The expected returns needs to be a ratio. +ve we expect an increase,
# -ve we expect a decrease. This will be feed into the model.
# Additonally, we need to take the log of the (returns+1), in an 
# attempt to remove the noise, to calc the cov matrix, other it will
# be ill conditioned.
returns=adj_close_df.pct_change()
returns = returns.drop(returns.index[0])
adj_close_df = adj_close_df.drop(adj_close_df.index[0])
returns.replace(0, 1e-6, inplace=True)
returns_log = np.log(1+returns)
returns_log.replace(0, 1e-6, inplace=True)

def Daily_return_cal(df, weights, i, N, Balance, Win_trades, Lose_trades):
    returnsX=((df.iloc[N+i,:]
              -df.iloc[N+i-1,:])
              /df.iloc[N+i-1,:])
    Total_ret=0
    
    Total_ret = np.dot(weights, returnsX)

    Balance+=Balance*(Total_ret)
    
    Total_ret=Total_ret*100
    
    if Total_ret>0:
        Win_trades+=1
    else:
        Lose_trades+=1
    
    return returnsX, Total_ret, Balance, Win_trades, Lose_trades

def Max_equity_drawdown_calc(Bal_vs_time_Part1):
    max_val_lookforward=max(Bal_vs_time_Part1)
    max_index_lookforward=Bal_vs_time_Part1.index(max_val_lookforward)
    min_val_lookforward=min(Bal_vs_time_Part1[max_index_lookforward+1:])

    min_val_lookbackward=min(Bal_vs_time_Part1)
    min_index_lookbackward=Bal_vs_time_Part1.index(min_val_lookbackward)
    max_val_lookbackward=max(Bal_vs_time_Part1[:min_index_lookbackward-1])

    lookforward_equity_drawdown=(max_val_lookforward-min_val_lookforward)/max_val_lookforward
    lookbackward_equity_drawdown=(max_val_lookbackward-min_val_lookbackward)/max_val_lookbackward

    equity_drawdown=round(max(lookforward_equity_drawdown, lookbackward_equity_drawdown)*100,1)
    return equity_drawdown

N_returns=30
N_covariance=400

Bal_vs_time_Part4=[]
Balance_Part4=10000
Returns_Part4=[]
Lose_trades_Part4=0
Win_trades_Part4=0

for i in range(0, len(df)-N_covariance-1):
    try:
        mu = (returns.iloc[N_covariance+i-N_returns:N_covariance+i,:]).mean()
        S = (returns_log.iloc[i:N_covariance+i,:]).cov()
        
        ef_Part4 = EfficientFrontier(mu, S, solver=SCS)
        weights_calc = ef_Part4.max_sharpe(risk_free_rate=(0.02/252))
        weights4 = np.array([value for value in weights_calc.values()])
        weights = np.clip(weights4, a_min=0, a_max=None)
        
        (returnsX_Part4,  
        Total_ret_Part4, 
        Balance_Part4, 
        Win_trades_Part4, 
        Lose_trades_Part4)=Daily_return_cal(
                                    adj_close_df, 
                                    weights4, 
                                    i, 
                                    N_covariance, 
                                    Balance_Part4,
                                    Win_trades_Part4,
                                    Lose_trades_Part4
                                    )
        
        Returns_Part4.append(Total_ret_Part4)
        Bal_vs_time_Part4.append(Balance_Part4)
    except Exception as e:
         print(f"Error occurred at index {i}: {str(e)}")
         
plt.hist(Returns_Part4, bins=10, edgecolor='black')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Histogram of daily returns')
mean = sum(Returns_Part4) / len(Returns_Part4)
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.legend()
plt.show()

plt.hist(Returns_Part4, bins=20, edgecolor='black', range=(-1.0, 1.0))
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Histogram of daily returns (Zoomed)')
mean = sum(Returns_Part4) / len(Returns_Part4)
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.legend()
plt.show()

plt.plot(Bal_vs_time_Part4)
plt.xlabel('Time (trading days: 2019-11-22 - 2023-05-31)')
plt.ylabel('Balance ($USD$)')
plt.title('Balance Over Time')
plt.show()

print('Number of winning trades: ', Win_trades_Part4)
print('Number of losing trades: ', Lose_trades_Part4)
print('Total number of trades made: ', (Win_trades_Part4+Lose_trades_Part4))
equity_drawdown=Max_equity_drawdown_calc(Bal_vs_time_Part4)
print('Maximum equity drawdown for the whole portfolio: ', equity_drawdown,
      ' %')
print('Maximum equity drawback from a single trade: ',
     round(min(Returns_Part4),1), 'as a percentage of equity invested.')