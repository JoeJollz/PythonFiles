# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:48:11 2023

@author: jrjol
"""

import alpaca_trade_api as api
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

import pandas as pd
import numpy as np
import statistics as s
#from sklearn.linear_model import LinearRegression

api = REST(
    key_id='PK3TY1J6MQKONR6CFHCU',
    secret_key='QIbzNGaZsDEhMTLRycKlJlHEksBMnZaZFXIgdMnf', 
    base_url='https://data.alpaca.markets/v2',
    api_version='v2',
)



df=api.get_bars('TSLA', TimeFrame(10, TimeFrameUnit.Minute), '2017-01-01', '2018-01-05').df


df.reset_index().plot(kind='scatter', x='timestamp', y='close')
df=df.iloc[::-1]


All_Average_PnL=[0]
Datapoint_range=[0]

### Defining Useful functions ####

def lin_reg(x,y,n,band_width):
    sum_x_squared=sum([i **2 for i in x])
    sum_y_values=sum(y)
    sum_x_values=sum(x)
    sum_xy_values=sum(np.multiply(x,y))
    square_sum_of_x=sum(x)**2
    
    alpha=(sum_x_squared*sum_y_values-sum_x_values*sum_xy_values)/(n*sum_x_squared-square_sum_of_x)
    
    beta=(n*sum_xy_values-sum_x_values*sum_y_values)/(n*sum_x_squared-square_sum_of_x)
    
    return alpha, beta
    


### Loop begins ####
#for n in range(5,40,1):
    
#for n in range(10,1000,10):
    n=500
    i=df.shape[0]-n
    
    short_buy_price=0
    Revenue=0
    trade_counter=0
    
    while i>=1:
    
            #### Data extraction ####
       
        x=df.iloc[i:i+n,3]
        y=range(1,n+1)
        
        alpha, beta= lin_reg(x,y,n,0)
        sigma=s.stdev(x)
        
        predicated_value_today=beta*302+alpha
        todays_value=df.iloc[i,3]
        yesterday_value=df.iloc[i+1,3]
        
        ### OPEN SHORT POSITION ###
        if todays_value>predicated_value_today and todays_value>=(predicated_value_today+sigma) and todays_value<yesterday_value:
            short_buy_price=todays_value
       
        if todays_value<predicated_value_today and todays_value<=(predicated_value_today-sigma) and todays_value>yesterday_value and short_buy_price>0:
            short_close_price=todays_value
           
            made_from_deal=short_buy_price-short_close_price
            Revenue=Revenue+made_from_deal
           
            trade_counter+=1
        
        if todays_value>1.05*short_buy_price:
            short_close_price=todays_value
           
            made_from_deal=short_buy_price-short_close_price
            Revenue=Revenue+made_from_deal
        
            trade_counter+=1
           
        
       
       
       
       
        i-=1
    
    ### Post Processing ###
    Aver_PnL_per_trade=Revenue/trade_counter

    All_Average_PnL.append(Aver_PnL_per_trade)
    Datapoint_range.append(n)
    
    
       ##### Calc the upper and lower linear regression lines #####