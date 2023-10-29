# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:16:27 2023

@author: jrjol
"""

from backtesting import Backtest, Strategy
import backtesting

backtesting.set_bokeh_output(notebook=False)
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
from sklearn.model_selection import train_test_split

df_Percentage_winners = pd.DataFrame()
df_number_of_trades = pd.DataFrame()
df_Maximum_eq_drawdowns_of_Port = pd.DataFrame()
df_worst_trades = pd.DataFrame()
df_Average_ROI_per_trade = pd.DataFrame()
df_Account_value = pd.DataFrame()


lookback_start=5
lookback_end=30
k_start=0.7
k_end=3
test_range=1
symbol="WMT"
ori_data = yf.download(symbol, start="2010-01-01", end="2017-01-01")
ori_data = ori_data.drop(columns=['Adj Close'])

long_states = [-4,-3]
long_odd_states = [ -2, -1, 1, 2, 3, 4]
short_states = [3, 4]
short_odd_states = [-4, -3, -2, -1, 1, 2]


def get_band_fast(yt, k, sigma):
    delta = k*sigma

    if yt > 0:
        # Uptrend Bands
        if yt <=   delta: return 1
        if yt <= 2*delta: return 2
        if yt <= 3*delta: return 3
        if yt <= 4*delta: return 4
        else: return 4

    elif yt < 0:
        # Uptrend Bands
        if yt >=   -delta: return -1
        if yt >= -2*delta: return -2
        if yt >= -3*delta: return -3
        if yt >= -4*delta: return -4
        else: return -4

    else:
        # Error
        return 0

def Get_States(df, look_back, k):
    """
        df: Dataframe of stocks Closing
        l : Lookback length

        linear_states: If you want states like {-4,-3,-2,-1,1,2,3,4} (False) or {0,1,2,3,4,5,6,7}
    """


    df["xt"] = df["Close"].pct_change() * 100
    df["std"] = df["xt"].rolling(look_back).std()
    df = df.dropna()

    # Calculate y_t
    #   1. set first y_t to 1
    #   2. do piecewise function on df

    
    # First 2 you can guarentee you see no trend, since trend must be 2 items or more
    df["yt"] = 0
    df.at[0, "yt"] = 1 
    df["yt"].iloc[1] = df["xt"].iloc[1]

    # Put 2 elements of close price
    prev = [*df["Close"].iloc[0:2]]
    
    # Store last yt each time (quicker than index 1 back)
    prev_yt = df["yt"].iloc[0]

    for indx,*row in df[2:].itertuples():
        # Check last two els including this
        if (prev[0] >= prev[1] >= row[3]) or (prev[0] <= prev[1] <= row[3]):
            # Set as previous yt * cur xt
            yt = prev_yt + row[6]
        else:
            # Set as just xt
            yt = row[6]

        # Set yt
        df.loc[indx, "yt"] = yt

        # ---- Update History Variables ----
        # Update yt
        prev_yt = yt
        
        # Append this Close to history
        # Trim list to 2 items
        prev.append(row[3])
        if (len(prev) > 2):
            del prev[0]

    # ---- Set states ----
    df["states"] = df.apply(lambda row: get_band_fast(row["yt"], k, row["std"]), axis=1)
    
    return df

def rsi(df, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi_values = ma_up / ma_down
    rsi_values = 100 - (100 / (1 + rsi_values))
    
    rsi_series = pd.Series(rsi_values, name='RSI')  # Change the name of the Series
    
    return rsi_series



RSI=rsi(ori_data)
RSI.index = ori_data.index
ori_data = pd.concat([ori_data, RSI], axis=1)
ori_data = ori_data.dropna()


def Max_equity_drawdown_calc(Account_Balance_Over_Time):
    if not Account_Balance_Over_Time:
        return 0
    else:
        max_val_lookforward=max(Account_Balance_Over_Time)
        max_index_lookforward=Account_Balance_Over_Time.index(max_val_lookforward)
        min_val_lookforward=min(Account_Balance_Over_Time[max_index_lookforward:])
    
        min_val_lookbackward=min(Account_Balance_Over_Time)
        if min_val_lookbackward==100:
            return 0
        else:
            
            min_index_lookbackward=Account_Balance_Over_Time.index(min_val_lookbackward)
            max_val_lookbackward=max(Account_Balance_Over_Time[:min_index_lookbackward])
    
        lookforward_equity_drawdown=(max_val_lookforward-min_val_lookforward)/max_val_lookforward
        lookbackward_equity_drawdown=(max_val_lookbackward-min_val_lookbackward)/max_val_lookbackward
    
        equity_drawdown=round(max(lookforward_equity_drawdown, lookbackward_equity_drawdown)*100,1)
        return equity_drawdown




for look_back in range(lookback_start,lookback_end,5):
    All_P_winners=[]
    All_trades=[]
    All_equity_drawdowns=[]
    All_average_rets=[]
    All_Final_Account_Values=[]
    All_single_worst_trades=[]
    row_names=[]
    for k in np.arange(k_start,k_end,0.1):
        k=round(k,2)
        data = Get_States(ori_data, look_back, k)
        Buy_and_Sell_data= pd.Series([0] * len(data), name='Overview')
        Buy_and_Sell_data.index = data.index
                
        open_position=0
        Account_Bal=100
        buy_long=0
        sell_long=0
        SLoss=0
        CMax=0
        Account_Balance_Over_Time=[]
        ROIs=[]
        lose_trades=0
        win_trades=0
        trades=1
        buy_dates=[]
        sell_dates=[]
        for I in range(2,len(data)):
            # if data.iloc[I-1]['Close']>CMax:
            #     SLoss*=1+(data.iloc[I-1]['Close']-CMax)/data.iloc[I-1]['Close']
            #     CMax=data.iloc[I-1]['Close']
            if data.iloc[I-2]['states'] in long_states and data.iloc[I-1]['states'] in long_odd_states and open_position==0:
                buy_long=data.iloc[I]['Open']*(1.002)
                open_position=1
                Buy_and_Sell_data.iloc[I-1]=1
                SLoss=buy_long*0.99
                CMax=buy_long
            # elif data.iloc[I-2]['states'] in short_states and data.iloc[I-1]['states'] in short_odd_states and open_position==1:
            #     sell_long = data.iloc[I]['Open']
            #     open_position=0
            #     Buy_and_Sell_data.iloc[I-1]=-1
            #     ROI=(sell_long-buy_long)/buy_long
            #     Account_Bal*=(1+ROI)
            #     ROIs.append(ROI)
            #     open_position=0
            #     sell_index=data.index[I]
            #     sell_dates.append(sell_index)
            #     if ROI>0:
            #         win_trades+=1
            #         trades+=1
            #     else:
            #         lose_trades+=1
            #         trades+=1
            # if data.iloc[I-1]['states'] in long_states and data.iloc[I-1]['RSI']<30 and open_position==0:
            #     buy_long=data.iloc[I]['Open']*(1.002)
            #     open_position=1
            #     Buy_and_Sell_data.iloc[I-1]=1
                
            # elif data.iloc[I-1]['states'] in short_states and open_position==1:
            #     sell_long = data.iloc[I]['Open']
            #     open_position=0
            #     Buy_and_Sell_data.iloc[I-1]=-1
            #     ROI=(sell_long-buy_long)/buy_long
            #     Account_Bal*=(1+ROI)
            #     ROIs.append(ROI)
            #     open_position=0
            #     sell_index=data.index[I]
            #     sell_dates.append(sell_index)
            #     if ROI>0:
            #         win_trades+=1
            #         trades+=1
            #     else:
            #         lose_trades+=1
            #         trades+=1
            
            # elif data.iloc[I-1]['Close']>buy_long and data.iloc[I-1]['RSI']>65 and open_position==1:
            #     sell_long = data.iloc[I]['Open']
            #     open_position=0
            #     Buy_and_Sell_data.iloc[I-1]=-1
            #     ROI=(sell_long-buy_long)/buy_long
            #     Account_Bal*=(1+ROI)
            #     ROIs.append(ROI)
            #     open_position=0
            #     sell_index=data.index[I]
            #     sell_dates.append(sell_index)
            #     if ROI>0:
            #         win_trades+=1
            #         trades+=1
            #     else:
            #         lose_trades+=1
            #         trades+=1
                
            elif data.iloc[I-1]['Close']<SLoss and open_position==1:
                sell_long = data.iloc[I]['Open']
                open_position=0
                Buy_and_Sell_data.iloc[I-1]=-1
                ROI=(sell_long-buy_long)/buy_long
                Account_Bal*=(1+ROI)
                ROIs.append(ROI)
                open_position=0
                sell_index=data.index[I]
                sell_dates.append(sell_index)
                if ROI>0:
                    win_trades+=1
                    trades+=1
                else:
                    lose_trades+=1
                    trades+=1
            elif data.iloc[I-1]['Close']>1.02*buy_long and open_position==1:
                sell_long = data.iloc[I]['Open']
                open_position=0
                Buy_and_Sell_data.iloc[I-1]=-1
                ROI=(sell_long-buy_long)/buy_long
                Account_Bal*=(1+ROI)
                ROIs.append(ROI)
                open_position=0
                sell_index=data.index[I]
                sell_dates.append(sell_index)
                if ROI>0:
                    win_trades+=1
                    trades+=1
                else:
                    lose_trades+=1
                    trades+=1
                
                
            Account_Balance_Over_Time.append(Account_Bal)
        if not ROIs:
            P_winners=0
            Ave=0
            equity_drawdown=0
            Single_trade_equity_drawdown=0
            Final_account_value=1
        else:
            P_winners=win_trades/trades*100
            Ave=sum(ROIs)/len(ROIs)
            equity_drawdown=Max_equity_drawdown_calc(Account_Balance_Over_Time)
            Single_trade_equity_drawdown=min(ROIs)
            Final_account_value=Account_Bal/100
        All_P_winners.append(P_winners)
        All_trades.append(trades)
        All_equity_drawdowns.append(-equity_drawdown)
        All_average_rets.append(Ave)
        All_Final_Account_Values.append(Final_account_value)
        All_single_worst_trades.append(Single_trade_equity_drawdown)
        

        row_names.append(f'k={k}')
        
    df_Percentage_winners[f'Lookback {look_back}'] = All_P_winners
    df_number_of_trades[f'Lookback {look_back}'] = All_trades
    df_Maximum_eq_drawdowns_of_Port[f'Lookback {look_back}'] = All_equity_drawdowns
    df_worst_trades[f'Lookback {look_back}'] = All_single_worst_trades
    df_Average_ROI_per_trade[f'Lookback {look_back}'] = All_average_rets
    df_Account_value[f'Lookback {look_back}'] = All_Final_Account_Values
 
df_Percentage_winners.index = row_names
df_number_of_trades.index = row_names
df_Maximum_eq_drawdowns_of_Port.index = row_names
df_worst_trades.index = row_names
df_Average_ROI_per_trade.index = row_names
df_Account_value.index = row_names   
            
                
data = pd.concat([data, Buy_and_Sell_data], axis=1)
data = data.dropna()



class Markov(Strategy):
    open_position=0
    def init(self):
        # self.markov = state_series
        pass
    def next(self):
        if self.data.Overview[-1] == 1:
            self.buy()
        if self.data.Overview[-1] ==-1:
            self.position.close()
            
        

bt = Backtest(data, Markov, commission=0.002, exclusive_orders=True)
stats = bt.run()
bt.plot()
                
        