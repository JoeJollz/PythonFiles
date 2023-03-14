# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:07:50 2023

@author: jrjol
"""

import alpaca_trade_api as api
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

import pandas as pd
import numpy as np
import statistics as s
import matplotlib.pyplot as plt
import math as m
#from sklearn.linear_model import LinearRegression

api = REST(
    key_id='PK3TY1J6MQKONR6CFHCU',
    secret_key='QIbzNGaZsDEhMTLRycKlJlHEksBMnZaZFXIgdMnf', 
    base_url='https://data.alpaca.markets/v2',
    api_version='v2',
)



df=api.get_bars('TSLA', TimeFrame(30, TimeFrameUnit.Minute), '2017-01-01', '2018-01-05').df


df.reset_index().plot(kind='scatter', x='timestamp', y='close')
df=df.iloc[::-1]


All_Average_PnL=[0]
Datapoint_range=[0]

### Defining Useful functions ####

def least_squares(x,y,n):
    sum_x_squared=sum([i **2 for i in x])
    sum_y_values=sum(y)
    sum_x_values=sum(x)
    sum_xy_values=sum(np.multiply(x,y))
    square_sum_of_x=sum(x)**2
    
    alpha=(sum_x_squared*sum_y_values-sum_x_values*sum_xy_values)/(n*sum_x_squared-square_sum_of_x)
    
    beta=(n*sum_xy_values-sum_x_values*sum_y_values)/(n*sum_x_squared-square_sum_of_x)
    
    return alpha, beta

def r_squared(x,y,n):
    sum_xy_values=sum(np.multiply(x,y))
    sum_x_values=sum(x)
    sum_y_values=sum(y)
    sum_x_squared=sum([i **2 for i in x])
    square_sum_of_x=sum(x)**2
    sum_y_squared=sum([i **2 for i in y])
    square_sum_of_y=sum(y)**2
    
    numerator=n*sum_xy_values-sum_x_values*sum_y_values
    denominator=m.sqrt((n*sum_x_squared-square_sum_of_x)*(n*sum_y_squared-square_sum_of_y))

    r=numerator/denominator
    r_square=r**2
    
    return r, r_square

df=df.between_time('14:30', '21:00')

a=10 #number of data points to predict the Std for vol
n=5 #number of data points to form the least squares
c=1 #One time step into the future we are predicting the value for.

i=df.shape[0]-a
stock_predications=[]
stock_actuals=[]
Residual_errors=[]
Trades=[]
Trade_number=[]
r_squares_calculated=[]
Revenue=0
buy_price=0
sell_price=0
made_from_trade=0
Counter=0
Profitable_trade=0
Loss_trade=0

while i>=1:
    y=df.iloc[i:i+a,3]
    std=s.stdev(y)
    
    y=df.iloc[i:i+n,3]
    y=y.iloc[::-1]
    x=range(1,n+1)
    
    alpha, beta =least_squares(x, y, n)
    r, r_square=r_squared(x,y,n)

    stock_predication=beta*(len(x)+1)+alpha
    # stock_actual=df.iloc[i-1,3]
    price_today=df.iloc[i,3]
    
    if stock_predication>price_today:
        buy_price=price_today
    
    i_tommorrow=i-1
    
    if i_tommorrow<i and buy_price>0:
        sell_price=df.iloc[i-1,3]
        
        made_from_trade=sell_price-buy_price
        Revenue=Revenue+made_from_trade
        Counter+=1
        
        Trades.append(made_from_trade)
        Trade_number.append(Counter)
        buy_price=0
        
        if made_from_trade>0:
            Profitable_trade+=1
        else:
            Loss_trade+=1
    
    
    # Residual_error=abs(stock_predication-stock_actual)
    
    
        stock_predications.append(stock_predication)
        r_squares_calculated.append(r_square)
    # stock_actuals.append(stock_actual)
    # Residual_errors.append(Residual_error)
    
    i-=1

Ave_PnL_Per_Trade=Revenue/Counter
Success_rate=Profitable_trade/(Loss_trade+Profitable_trade)*100

df_results=pd.DataFrame({'Trade Number': Trade_number, 'PnL per trade $USD$': Trades})

