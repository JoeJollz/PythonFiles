# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:10:11 2023

@author: jrjol
"""

from sklearn.linear_model import LassoCV
import alpaca_trade_api as alpaca

alpaca_api = alpaca.REST(
    key_id='PK3TY1J6MQKONR6CFHCU',
    secret_key='QIbzNGaZsDEhMTLRycKlJlHEksBMnZaZFXIgdMnf', 
    base_url='https://data.alpaca.markets/v2',
    api_version='v2',
)

alpaca_df = alpaca_api.get_bars('SPY', alpaca.TimeFrame(30, alpaca.TimeFrameUnit.Minute), '2015-01-01', '2023-01-05').df
alpaca_df = alpaca_df.between_time('14:30', '21:00')

df_training = alpaca_df.iloc[0:18013]

X = df_training[['open', 'high', 'low', 'vwap', 'trade_count', 'volume']]
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