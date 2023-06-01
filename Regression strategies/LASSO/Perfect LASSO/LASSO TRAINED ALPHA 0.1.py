# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:28:19 2023

@author: jrjol
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import alpaca_trade_api as alpaca
import matplotlib.pyplot as plt


alpaca_api = alpaca.REST(
    key_id='PK3TY1J6MQKONR6CFHCU',
    secret_key='QIbzNGaZsDEhMTLRycKlJlHEksBMnZaZFXIgdMnf', 
    base_url='https://data.alpaca.markets/v2',
    api_version='v2',
)

alpaca_df = alpaca_api.get_bars('SPY', alpaca.TimeFrame(30, alpaca.TimeFrameUnit.Minute), '2015-01-01', '2023-01-05').df
alpaca_df = alpaca_df.between_time('14:30', '21:00')

df_training = alpaca_df.iloc[0:18013]
df_test = alpaca_df.iloc[18013:25013]

X_train = df_training.drop(['close'], axis=1)
Y_train = df_training['close']

X_test = df_test.drop(['close'], axis=1)
Y_test = df_test['close']

lasso = Lasso(alpha=0.1, max_iter=1000000)

lasso.fit(X_train, Y_train)

y_pred = lasso.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)

coefficients = pd.DataFrame({'Features': X_train.columns, 'Coefficients': lasso.coef_})
print(coefficients)

plt.scatter(Y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Lasso Regression: Actual vs Predicted values")
plt.show()

X_pred=range(1,df_test.shape[0]+1)

fig, ax = plt.subplots()

ax.plot(X_pred, y_pred, label='Predicted Close')
ax.plot(X_pred, Y_test, label='Actual Close')
ax.set_xlabel('Time steps: 30 minutes')
ax.set_ylabel('Closing Values SPY $$$')
ax.set_title('Predicted Close vs Actual Close')
ax.legend()

plt.show()

r_squared = r2_score(Y_test, y_pred)
print("R-squared (R^2):", r_squared)

corr_coef = np.corrcoef(Y_test, y_pred)[0, 1]
print("Correlation Coefficient (R):", corr_coef)

residuals = Y_test - y_pred
rss = np.sum(residuals**2)
n = len(Y_test)
k = 6
aic = n * np.log(rss/n) + 2 * k
