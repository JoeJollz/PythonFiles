# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:40:17 2023

@author: jrjol
"""

import alpaca_trade_api as api
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

import pandas as pd
import numpy as np
import statistics as s
import matplotlib.pyplot as plt
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

df=df.between_time('14:30', '21:00')

a=10 #number of data points to predict the Std for vol
n=5 #number of data points to form the least squares
c=1 #One time step into the future we are predicting the value for.

i=df.shape[0]-a
stock_predications=[]
stock_actuals=[]
Residual_errors=[]

while i>=1:
    y=df.iloc[i:i+a,3]
    std=s.stdev(y)
    
    y=df.iloc[i:i+n,3]
    y=y.iloc[::-1]
    x=range(1,n+1)
    
    alpha, beta =least_squares(x, y, n)

    stock_predication=beta*(len(x)+1)+alpha
    stock_actual=df.iloc[i-1,3]
    
    Residual_error=abs(stock_predication-stock_actual)
    
    
    stock_predications.append(stock_predication)
    stock_actuals.append(stock_actual)
    Residual_errors.append(Residual_error)
    
    i-=1

df_results=pd.DataFrame({'Predicted stock value': stock_predications, 'Actual stock value': stock_actuals, 'Residual Error': Residual_errors})

x_axis=np.linspace(0,50)
actual=stock_actuals[:50]
Predicted=stock_predications[:50]

fig, ax = plt.subplots()
# plot the data on the first axis
ax.plot(x_axis, actual, label='Actual stock price')
ax.set_xlabel('Time step (30 mins)')
ax.set_ylabel('Actual stock price')

# create a second y-axis object and plot the second dependent variable
ax2 = ax.twinx()
ax2.plot(x_axis, Predicted, color='red', label='Predicted stock price')
ax2.set_ylabel('Predicted stock price')

# add a legend and show the plot
fig.legend()
plt.show()
