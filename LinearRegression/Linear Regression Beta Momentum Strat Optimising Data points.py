# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:45:28 2023

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
All_n_values=[0]
All_average_PnL=[0]
All_counter=[0]


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

for n in range(150,1550,100):

    i=df.shape[0]-n
    short_buy_price=0
    long_buy_price=0
    Revenue=0
    counter=0
    
    
    while i>=1:
         x=df.iloc[i:i+n,3]
         y=range(1,n+1)
         
         todays_price=df.iloc[i,3]
            
         alpha, beta= lin_reg(x,y,n,0)
         
         ## UPWARDS MOMENTUM - BUY LONG ###
         if beta>0 and long_buy_price==0:
             long_buy_price=todays_price
        
        ## CLOSE THE LONG POSITION ##
         if beta<0 and long_buy_price>0:
             close_long_price=todays_price
            
             made_from_deal=close_long_price-long_buy_price
             Revenue=Revenue+made_from_deal
             counter+=1
             
             long_buy_price=0
         i-=1
    
    Average_PnL_per_trade=Revenue/counter

    All_n_values.append(n)
    All_average_PnL.append(Average_PnL_per_trade)
    All_counter.append(counter)
    
df_results=pd.DataFrame({'How many data points to detemine beta':All_n_values,'Average PnL per Trade': All_average_PnL,'How many trades made':All_counter})



     