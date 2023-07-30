# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:10:09 2023

@author: jrjol
"""


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="max_sharpe transforms the optimization problem")
import finnhub
import requests
from bs4 import BeautifulSoup
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from cvxpy import ECOS, SCS

# Send a GET request to the Wikipedia page
url = 'https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies'
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table containing the S&P 500 symbols
table = soup.find('table', {'id': 'constituents'})

# Extract the ticker symbols from the 'Symbol' column
tickers = []
for row in table.find_all('tr')[1:]:
    columns = row.find_all('td')
    ticker = columns[0].text.strip()
    tickers.append(ticker)

# Print the ticker symbols
print(tickers)



# Fetch historical data for S&P 500 tickers using yfinance
df = yf.download(tickers, start="2017-11-22", end="2023-05-31")['Adj Close']
# Assuming df is your DataFrame
df = df.dropna(axis=1)

original_num_col=df.shape[1]

def Tech(df, i, MA_one, MA_two):
    # df[f'Short_SMA_{i}']=df.iloc[:,i].rolling(window=200).mean()
    # df[f'Long_SMA_{i}']=df.iloc[:,i].rolling(window=50).mean()
    S=pd.Series(df.iloc[:,i].rolling(window=200).mean(), name=f'Short_SMA_{i}')
    L=pd.Series(df.iloc[:,i].rolling(window=50).mean(), name=f'Long_SMA_{i}')
    df=pd.concat([df,S], axis=1)
    df=pd.concat([df,L], axis=1)
    return df

MA_one=20
MA_two=50
for i in range(0,df.shape[1]):
    df=Tech(df, i, MA_one, MA_two)

df=df.dropna(axis=0)

Num_rows=df.shape[0]

df_open = yf.download(tickers, start="2017-11-22", end="2023-05-31")['Open']
# Assuming df is your DataFrame
df_open = df_open.dropna(axis=1)
df_open=df_open.tail(Num_rows)

spy_data = yf.download("SPY", start="2017-11-22", end="2023-05-31")['Adj Close']
spy_data=spy_data.tail(Num_rows)


def Golden_Cross_Over(df, df_open, I, original_num_col):
    C=0
    for X in range(original_num_col, (df.shape[1]), 2):
        tday_short_SMA=df.iloc[I,X]
        yday_short_SMA=df.iloc[I-1,X]
        tday_long_SMA=df.iloc[I,X+1]
        yday_long_SMA=df.iloc[I-1,X+1]
        if tday_short_SMA>tday_long_SMA and yday_short_SMA<yday_long_SMA:
            index=X-original_num_col-C
            buy_price=df_open.iloc[I+1,index]
            open_position=1
            Stoploss=0.97*buy_price
            TPLevel=buy_price*1.03
            return buy_price, open_position, index, C, X, Stoploss, TPLevel
        C+=1
    return 0,0,0,0,0,0,0

def Red_Cross_Over(df, df_open, I, original_num_col, buy_price, index, C):
    X=index+original_num_col+C
    tday_short_SMA=df.iloc[I,X]
    yday_short_SMA=df.iloc[I-1,X]
    tday_long_SMA=df.iloc[I,X+1]
    yday_long_SMA=df.iloc[I-1,X+1]
    revenue=0
    open_position=1
    if tday_short_SMA<tday_long_SMA and yday_short_SMA>yday_long_SMA:
        sell_price=df_open.iloc[I+1,index]
        revenue=sell_price-buy_price
        ROI=(sell_price-buy_price)/buy_price
        open_position=0
        return revenue, open_position, ROI
    return 0, 1, 0

def Stop_Loss(df, df_open, I, buy_price, index, Stoploss, C):
    current_price=df.iloc[I,index]
    if current_price<Stoploss:
        sell_price=df_open.iloc[I+1,index]
        revenue=sell_price-buy_price
        ROI=(sell_price-buy_price)/buy_price
        open_position=0
        Stoploss=0
        return revenue, open_position, Stoploss, ROI
    
    elif current_price>=df.iloc[I-1,index]:
        Stoploss=current_price*0.98
        revenue=0
        ROI=0
        open_position=1
        return revenue, open_position, Stoploss, ROI
    elif current_price<df.iloc[I-1,index]:
        Stoploss=Stoploss
        revenue=0
        ROI=0
        open_position=1
        return revenue, open_position, Stoploss, ROI
    
def Take_Profit(df, df_open, I, buy_price, TPLevel):
    current_price=df.iloc[I,index]
    if current_price>TPLevel:
        sell_price=df_open.iloc[I+1,index]
        revenue=sell_price-buy_price
        ROI=(sell_price-buy_price)/buy_price
        open_position=0
        return revenue, open_position, ROI
    else:
        return 0, 1, 0
    
days_held=0
Holding_periods=[]
ROIs=[]
bps=[]
open_position=0
Total_Rev=0
revenue=0
Trades=0
Revenue_from_Trades=[]
Bal_Part1=100
Updated_Bal_Part1=[]
Bal_Part2=100
Updated_Bal_Part2=[]

for I in range(1, len(df)-1):
    if open_position==0:    
        buy_price, open_position, index, C, X, Stoploss, TPLevel=Golden_Cross_Over(df, df_open, I, original_num_col)
    if open_position!=0:
        days_held+=1
    if open_position !=0:
        revenue, open_position, ROI = Red_Cross_Over(df, df_open, I, original_num_col, buy_price, index, C)
    if open_position !=0:
        revenue, open_position, Stoploss, ROI = Stop_Loss(df, df_open, I, buy_price, index, Stoploss, C)
    # if open_position !=0:
    #     revenue, open_position, ROI = Take_Profit(df, df_open, I, buy_price, TPLevel)
    bps.append(buy_price)
    Bal_Part1*=(1+ROI)
    Updated_Bal_Part1.append(Bal_Part1)
    Bal_Part2*=(1+(spy_data.iloc[I]-spy_data.iloc[I-1])/spy_data.iloc[I-1])
    Updated_Bal_Part2.append(Bal_Part2)
    if revenue!=0:
        Total_Rev+=revenue
        Revenue_from_Trades.append(revenue)
        Holding_periods.append(days_held)
        ROIs.append(ROI)
        ROI=0
        revenue=0
        days_held=0
        Trades+=1
        


plt.plot(Updated_Bal_Part1, label='Algorithm')
plt.plot(Updated_Bal_Part2, label='S&P500')

# Adding labels and legend
plt.xlabel('Trading Days (22/11/2017 - 31/05/2023')
plt.ylabel('Account Value')
plt.legend()

# Display the plot
plt.show()

    