# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:03:41 2023

@author: jrjol
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import alpaca_trade_api as alpaca
import matplotlib.pyplot as plt
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


X = df_training[['open', 'high', 'low', 'vwap']]
y = df_training['close']


lasso_cv = LassoCV(cv=5)


lasso_cv.fit(X, y)

print('Lasso coefficients:', lasso_cv.coef_)
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

def Input_PREDICTIONS(df_training, i):
    def lin_reg(x,z,n):
        sum_x_squared=sum([a **2 for a in x])
        sum_y_values=sum(z)
        sum_x_values=sum(x)
        sum_xy_values=sum(np.multiply(x,z))
        square_sum_of_x=sum(x)**2
        
        alpha=(sum_x_squared*sum_y_values-sum_x_values*sum_xy_values)/(n*sum_x_squared-square_sum_of_x)
        
        beta=(n*sum_xy_values-sum_x_values*sum_y_values)/(n*sum_x_squared-square_sum_of_x)
        
        return alpha, beta
    predictions=[0]
    predictions=[]
    
    size=df_training.shape[0]
    
    
    for p in range(0,4):
        x=range(1,3)
        z=df_training.iloc[size-1:size+1,p].values.tolist()
        n=2
        
        alpha , beta = lin_reg(x, z, n)
        
        prediction=beta*3+alpha
        
        predictions.append(prediction)
    
    open_pred=predictions[0]
    high_pred=predictions[1]
    low_pred=predictions[2]
    vwap_pred=predictions[3]
   
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
    
def LASSO_REGRESSOR(open_coefficient, high_coefficient, 
                    low_coefficient, vol_coefficient, 
                    alpha, open_pred, high_pred, low_pred, 
                    vwap_pred
                    ):
    
    Regul=open_coefficient+high_coefficient+low_coefficient+vwap_coefficient
    pred_close=(open_coefficient*open_pred
                +high_coefficient*high_pred
                +low_coefficient*low_pred
                +vwap_coefficient*vwap_pred
                +alpha*Regul
                )
    return pred_close



future_iterations=alpaca_df.shape[0]-df_training.shape[0]

all_y_pred=[]
all_pred_closes=[]
all_absoulte_errors=[]
Data_of_Percentage_winners=[]
Data_of_all_trades=[]
actual_closes=[]



for i in range(2,future_iterations-1):
    
    df_training=alpaca_df.iloc[i:18013+i]
    
    actual_future_close=all_actual_close.iloc[i]
    
    current_close=alpaca_df.iloc[18013+i,3]
    
    open_pred, high_pred, low_pred, vwap_pred=Input_PREDICTIONS(df_training, i)
    
    pred_close=LASSO_REGRESSOR(open_coefficient, high_coefficient, 
                               low_coefficient, vwap_coefficient, 
                               alpha, open_pred, high_pred, low_pred,
                               vwap_pred
                               )
    pred_close=pred_close/5
    
    all_pred_closes.append(pred_close)
    actual_closes.append(actual_future_close)
    
    
    
    interval_counter+=1
    
    if interval_counter==14:
        open_coefficient, high_coefficient, low_coefficient, vwap_coefficient, alpha=LASSO_COEFF_CALCULATOR(df_training)
        interval_counter=0
    
    n_params = 4
    absoulte_error=abs(pred_close-actual_future_close)
    all_absoulte_errors.append(absoulte_error)
    Sum_of_errors=Sum_of_errors+absoulte_error
    MSE=Sum_of_errors/(6999)
    aic_value = 2 * n_params + interval_counter * np.log(Sum_of_errors / i)

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


x = range(0,len(all_pred_closes))

fig, ax1 = plt.subplots()

ax1.plot(x, actual_closes, color='blue', label='True close')
ax1.set_xlabel('x')
ax1.set_ylabel('', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()

ax2.plot(x, all_pred_closes, color='red', label='exp(x)')
ax2.set_ylabel('exp(x)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

plt.show()

r_squared = r2_score(actual_closes, all_pred_closes)
print("R-squared (R^2):", r_squared)

corr_coef = np.corrcoef(actual_closes, all_pred_closes)[0, 1]
print("Correlation Coefficient (R):", corr_coef)

actual_closes_arr = np.array(actual_closes)
all_pred_closes_arr = np.array(all_pred_closes)
residuals = actual_closes_arr - all_pred_closes_arr
rss = np.sum(residuals**2)
n = len(actual_closes)
k = 6
aic = n * np.log(rss/n) + 2 * k

    


    
    