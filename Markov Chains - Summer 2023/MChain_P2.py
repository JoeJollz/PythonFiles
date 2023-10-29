# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 08:13:03 2023

@author: jrjol
"""

"""
Following .py file is related to Markov Chains. In this script, only state transition probability matirices have been defined.
We have 2 state transition probability matricies, one applicable for market uptrends, the other applicable for market downtrends.
Market uptrend is defined when the Simple Moving Average (SMA) (with a lower lookback period) is priced above the SMA (with a greater lookback 
period), and vice versa.

From this, we can view the state transition probilities, and capitalise on the probability of state transition to make a trade.

A single state is represented by a single candle (in this case, a single day), a down state is defined as a day with negaitve returns,
in contrast, an up state is defined as a day with positive returns. 

"""


import yfinance as yf
import numpy as np

from sklearn.model_selection import train_test_split

symbol = "SPY"

spy_data = yf.download(symbol, start="2015-01-01", end="2023-05-31")
spy_data.reset_index(inplace=True)
spy_data.set_index('Date', inplace=True)

spy_data=spy_data[['Open', 'Close']]

Daily_residual=(spy_data['Close']-spy_data['Open'])/spy_data['Open']

std=np.std(Daily_residual)

SSMA=15
LSMA=45

spy_data['SMA_short']=spy_data.iloc[:,0].rolling(SSMA).mean()
spy_data['SMA_long']=spy_data.iloc[:,0].rolling(LSMA).mean()

train_data, test_data = train_test_split(spy_data, test_size=0.3)

UT_Neg_Neg=0
UT_Neg_Pos=0
UT_Pos_Neg=0
UT_Pos_Pos=0

DT_Neg_Neg=0
DT_Neg_Pos=0
DT_Pos_Neg=0
DT_Pos_Pos=0

DT_one=0
DT_two=0
UT_one=0
UT_two=0

for I in range(LSMA,len(train_data)):
    if spy_data.iloc[I,2]>spy_data.iloc[I,3]:  ## If there is an uptrend, based on the moving average cross over.
        if Daily_residual.iloc[I]>0 and Daily_residual.iloc[I-1]>0:
            UT_Pos_Pos+=1
            UT_one+=1
            
        if Daily_residual.iloc[I]<0 and Daily_residual.iloc[I-1]>0:
            UT_Pos_Neg+=1
            UT_one+=1
        if Daily_residual.iloc[I]>0 and Daily_residual.iloc[I-1]<0:
            UT_Neg_Pos+=1
            UT_two+=1
        if Daily_residual.iloc[I]<0 and Daily_residual.iloc[I-1]<0:
            UT_Neg_Neg+=1
            UT_two+=1
    if spy_data.iloc[I,2]<spy_data.iloc[I,3]:  ## If there is a downtrend, based on the moving average cross over.         
    
        if Daily_residual.iloc[I]>0 and Daily_residual.iloc[I-1]>0:
            DT_Pos_Pos+=1
            DT_one+=1
        if Daily_residual.iloc[I]<0 and Daily_residual.iloc[I-1]>0:
            DT_Pos_Neg+=1
            DT_one+=1
        if Daily_residual.iloc[I]>0 and Daily_residual.iloc[I-1]<0:
            DT_Neg_Pos+=1
            DT_two+=1
        if Daily_residual.iloc[I]<0 and Daily_residual.iloc[I-1]<0:
            DT_Neg_Neg+=1
            DT_two+=1

PUT_Neg_Neg=UT_Neg_Neg/UT_two
PUT_Neg_Pos=UT_Neg_Pos/UT_two
PUT_Pos_Neg=UT_Pos_Neg/UT_one
PUT_Pos_Pos=UT_Pos_Pos/UT_one

PDT_Neg_Neg=DT_Neg_Neg/DT_two
PDT_Neg_Pos=DT_Neg_Pos/DT_two
PDT_Pos_Neg=DT_Pos_Neg/DT_one
PDT_Pos_Pos=DT_Pos_Pos/DT_one

