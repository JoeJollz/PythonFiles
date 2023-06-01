# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:57:25 2023

@author: jrjol
"""

import threading
import numpy as np
import pandas as pd
import alpaca_trade_api as alpaca
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import concurrent.futures
from sklearn.linear_model import LassoCV



alpaca_api = alpaca.REST(
    key_id='PK3TY1J6MQKONR6CFHCU',
    secret_key='QIbzNGaZsDEhMTLRycKlJlHEksBMnZaZFXIgdMnf', 
    base_url='https://data.alpaca.markets/v2',
    api_version='v2',
)

alpaca_df = alpaca_api.get_bars('SPY', alpaca.TimeFrame(30, alpaca.TimeFrameUnit.Minute), '2015-01-01', '2023-01-05').df

alpaca_df = alpaca_df.between_time('14:30', '21:00')


df_training = alpaca_df.iloc[0:18013]
df_test=alpaca_df.iloc[18014:25013]
all_actual_close=df_test['close']

### I have selected on these features, this can be changed to include: volume, trade_count##
X = df_training[['open', 'high', 'low', 'vwap']]
y = df_training['close']


lasso_cv = LassoCV(cv=5)


lasso_cv.fit(X, y)


print('Lasso coefficients:', lasso_cv.coef_)
print('Lasso intercept:', lasso_cv.intercept_)
print('Best alpha:', lasso_cv.alpha_)

coef = lasso_cv.coef_
open_coefficient=coef[0]
high_coefficient=coef[1]
low_coefficient=coef[2]
vwap_coefficient=coef[3]
alpha=lasso_cv.alpha_

interval_counter=0
Sum_of_errors=0
buy_price=0
Revenue=0
winning_trades=0
losing_trades=0

def ARIMA_PREDICTIONS(df_training):

    
    open_future = concurrent.futures.Future()
    high_future = concurrent.futures.Future()
    low_future = concurrent.futures.Future()
    #volume_future = concurrent.futures.Future()
    #trade_count_future = concurrent.futures.Future()
    vwap_future = concurrent.futures.Future()

    
    thread1 = threading.Thread(target=lambda: open_future.set_result(auto_arima((df_training.iloc[:,0]), seasonal=False, trace=True, max_q=3, max_d=3, max_p=3)), args=())
    thread2 = threading.Thread(target=lambda: high_future.set_result(auto_arima((df_training.iloc[:,1]), seasonal=False, trace=True, max_q=3, max_d=3, max_p=3)), args=())
    thread3 = threading.Thread(target=lambda: low_future.set_result(auto_arima((df_training.iloc[:,2]),  seasonal=False, trace=True, max_q=3, max_d=3, max_p=3)), args=())
    #thread4 = threading.Thread(target=lambda: volume_future.set_result(auto_arima((df_training.iloc[:,4]),  seasonal=False, trace=True, max_q=3, max_d=3, max_p=3)), args=())
    #thread5 = threading.Thread(target=lambda: trade_count_future.set_result(auto_arima((df_training.iloc[:,5]),  seasonal=False, trace=True, max_q=3, max_d=3, max_p=3)), args=())
    thread6 = threading.Thread(target=lambda: vwap_future.set_result(auto_arima((df_training.iloc[:,6]), seasonal=False, trace=True, max_q=3, max_d=3, max_p=3)), args=())

    thread1.start()
    thread2.start()
    thread3.start()
    #thread4.start()
    #thread5.start()
    thread6.start()

    thread1.join()
    thread2.join()
    thread3.join()
    #thread4.join()
    #thread5.join()
    thread6.join()

    
    open = open_future.result()
    high = high_future.result()
    low = low_future.result()
    #volume = volume_future.result()
    #trade_count = trade_count_future.result()
    vwap = vwap_future.result()

    
    open_p, open_d, open_q = open.order
    high_p, high_d, high_q = high.order
    low_p, low_d, low_q = low.order
    #volume_p, volume_d, volume_q = volume.order
    #trade_count_p, trade_count_d, trade_count_q = trade_count.order
    vwap_p, vwap_d, vwap_q = vwap.order
    
    
    open_model_fit = ARIMA(df_training.iloc[:,0], order=(open_p,open_d,open_q), seasonal_order=(0,0,0,0)).fit()
    high_model_fit = ARIMA(df_training.iloc[:,1], order=(high_p,high_d,high_q), seasonal_order=(0,0,0,0)).fit()
    low_model_fit = ARIMA(df_training.iloc[:,2], order=(low_p,low_d,low_q), seasonal_order=(0,0,0,0)).fit()
    #volume_model_fit = ARIMA(df_training.iloc[:,4], order=(volume_p,volume_d,volume_q), seasonal_order=(0,0,0,0)).fit()
    #trade_count_model_fit = ARIMA(df_training.iloc[:,5], order=(trade_count_p,trade_count_d,trade_count_q), seasonal_order=(0,0,0,0)).fit()
    vwap_model_fit = ARIMA(df_training.iloc[:,6], order=(vwap_p,vwap_d,vwap_q), seasonal_order=(0,0,0,0)).fit()
    
    ####
    forecast_open = open_model_fit.forecast(steps=1)
    open_pred = forecast_open.iloc[0]
    
    forecast_high = high_model_fit.forecast(steps=1)
    high_pred = forecast_high.iloc[0]
    
    forecast_low = low_model_fit.forecast(steps=1)
    low_pred = forecast_low.iloc[0]
    
    #forecast_volume = volume_model_fit.forecast(steps=1)
    #volume_pred = forecast_volume.iloc[0]
    
    #forecast_trade_count = trade_count_model_fit.forecast(steps=1)
    #trade_count_pred = forecast_trade_count.iloc[0]
    
    forecast_vwap = vwap_model_fit.forecast(steps=1)
    vwap_pred = forecast_vwap.iloc[0]
    
    return open_pred, high_pred, low_pred, vwap_pred

def LASSO_COEFF_CALCULATOR(df_training):
    X = df_training[['open', 'high', 'low', 'vwap']]
    y = df_training['close']

    
    lasso_cv = LassoCV(cv=5)

    lasso_cv.fit(X, y)

    coef = lasso_cv.coef_
    open_coefficient=coef[0]
    high_coefficient=coef[1]
    low_coefficient=coef[2]
    vwap_coefficient=coef[3]
    alpha=lasso_cv.alpha_
    
    return open_coefficient, high_coefficient, low_coefficient, vwap_coefficient, alpha
    
def LASSO_REGRESSOR(open_coefficient, high_coefficient, low_coefficient, vwap_coefficient, alpha, open_pred, high_pred, low_pred, vwap_pred):
    pred_close=open_coefficient*open_pred+high_coefficient*high_pred+low_coefficient*low_pred+vwap_coefficient*vwap_pred+alpha*(open_coefficient+high_coefficient+low_coefficient+vwap_coefficient)
    return pred_close



future_iterations=alpaca_df.shape[0]-df_training.shape[0]

all_y_pred=[]
all_pred_closes=[]
all_absoulte_errors=[]
Data_of_Percentage_winners=[]
Data_of_all_trades=[]



for i in range(0,future_iterations-1):
    
    df_training=alpaca_df.iloc[i:18013+i]
    
    actual_future_close=all_actual_close.iloc[i]
    
    current_close=alpaca_df.iloc[18013+i,3]
    
    open_pred, high_pred, low_pred, vwap_pred=ARIMA_PREDICTIONS(df_training)
    
    pred_close=LASSO_REGRESSOR(open_coefficient, high_coefficient, low_coefficient, vwap_coefficient, alpha, open_pred, high_pred, low_pred, vwap_pred)
    
    interval_counter+=1
    
    if interval_counter==14:
        open_coefficient, high_coefficient, low_coefficient, vwap_coefficient, alpha=LASSO_COEFF_CALCULATOR(df_training)
        interval_counter=0
    
    n_params = 4
    absoulte_error=abs(pred_close-actual_future_close)
    all_absoulte_errors.append(absoulte_error)
    Sum_of_errors=Sum_of_errors+absoulte_error
    MSE=Sum_of_errors/(i+1)
    aic_value = 2 * n_params + interval_counter * np.log(Sum_of_errors / interval_counter)

    if pred_close>current_close and buy_price==0:
        buy_price=current_close
    
    if buy_price>0:
        sell_price=actual_future_close
        
        money_made=sell_price-buy_price
        buy_price=0
        
        Data_of_all_trades.append(money_made)
        
        Revenue=Revenue+money_made
        
        if money_made>0:
            winning_trades+=1
        else:
            losing_trades+=1
            
        Percentage_winners=winning_trades/(winning_trades+losing_trades)*100
        Data_of_Percentage_winners.append(Percentage_winners)
        
        Average_revenue_per_trade=Revenue/(winning_trades+losing_trades)
    
    
    print('How many iterations: ', interval_counter)
    print('Percentage of winners: ', Data_of_Percentage_winners)
    print('All trades: ', Data_of_all_trades)
    print('Current Revenue: ', Revenue)
    print('Average revuene per trade so far: ', Average_revenue_per_trade)
    print('Number of trades made: ', (winning_trades+losing_trades), '. Of these, ', winning_trades, ' are winners, and, ', losing_trades, ' are losing trades.')
    print('MSE: ', MSE)
    print('AIC value:', aic_value)
    


    
    