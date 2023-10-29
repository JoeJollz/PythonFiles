# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:04:00 2023

@author: jrjol
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import requests
import json

def checkindicator(url):
    r= requests.get(url)
    r = r.json()
    periods = r['series']['docs'][0]['period']
    values = r['series']['docs'][0]['value']
    dataset = r['series']['docs'][0]['dataset_name']
    indicators = pd.DataFrame(values,index=periods)
    indicators.columns = [dataset]
    return indicators

unemployment_rate = checkindicator("https://api.db.nomics.world/v22/series/BLS/ln/LNS13327707?observations=1")
PPI = checkindicator("https://api.db.nomics.world/v22/series/OECD/DP_LIVE/USA.PPI.TOT_MKT.IDX2015.M?observations=1")
CPI = checkindicator("https://api.db.nomics.world/v22/series/IMF/CPI/M.US.PCPI_IX?observations=1")
interest_rate = checkindicator("https://api.db.nomics.world/v22/series/OECD/MEI/USA.IRSTCI01.ST.M?observations=1")

new_unemployment_rate = unemployment_rate.loc[unemployment_rate.index>='1999-01']
print(new_unemployment_rate)

# PPI with 2015 = 100
# https://db.nomics.world/OECD/DP_LIVE/USA.PPI.TOT_MKT.IDX2015.M
# PPI = checkindicator("https://api.db.nomics.world/v22/series/OECD/DP_LIVE/USA.PPI.TOT_MKT.IDX2015.M?observations=1")

new_PPI = PPI.loc[PPI.index>='1999-01']
print(new_PPI)
print(new_PPI.columns)

# CPI
# https://db.nomics.world/IMF/CPI/M.US.PCPI_IX
# CPI = checkindicator("https://api.db.nomics.world/v22/series/IMF/CPI/M.US.PCPI_IX?observations=1")
new_CPI = CPI.loc[CPI.index>='1999-01']
print(new_CPI)

#interest rate
# https://db.nomics.world/OECD/MEI/USA.IRSTCI01.ST.M
interest_rate = checkindicator("https://api.db.nomics.world/v22/series/OECD/MEI/USA.IRSTCI01.ST.M?observations=1")
new_interest_rate = interest_rate.loc[interest_rate.index>='1999-01']
print(new_interest_rate)

#normalized GDP  this is the best I can get so far
# https://db.nomics.world/OECD/MEI_CLI/LORSGPNO.USA.M
normalized_GDP = checkindicator('https://api.db.nomics.world/v22/series/OECD/MEI_CLI/LORSGPNO.USA.M?observations=1')
new_normalized_GDP = normalized_GDP.loc[normalized_GDP.index >= '1999-01']
print(new_normalized_GDP)

# new_CPI['PPI'] = new_PPI['OECD Data Live dataset']

combined_df = new_CPI
combined_df['PPI'] = new_PPI['OECD Data Live dataset']
combined_df['Unemployment_rate'] = new_unemployment_rate['Labor Force Statistics including the National Unemployment Rate']
combined_df['interest_rate'] = new_interest_rate['Main Economic Indicators Publication']
combined_df['normalized_GDP'] = new_normalized_GDP['Composite Leading Indicators (MEI)']

print(combined_df)

spy_data = yf.download("^GSPC", start="1999-01-01", end="2023-09-01", interval="1mo")
'''2023-08-01 (2023-08) represents the Open High Low Close Volume for '''
spy_data = spy_data.reindex(combined_df.index)
print(spy_data.head())
print(len(spy_data))

spy_data = spy_data.reindex(combined_df.index)
Main_df = pd.concat([combined_df, spy_data], axis=1)
Main_df = Main_df.dropna()


print(Main_df)

feature_scaler=MinMaxScaler()
scaled_data=feature_scaler.fit_transform(Main_df)
min_values = feature_scaler.data_min_
max_values = feature_scaler.data_max_


scaling_factors_df = pd.DataFrame({'Feature': Main_df.columns,
                                   'Min Value': min_values,
                                   'Max Value': max_values})

print(scaling_factors_df)

print(scaled_data)
print(len(scaled_data))

look_back=8

# def create_dataset(data, look_back):
#     X, y = [], []
#     for i in range(look_back, len(data)-1):
#         X.append(data[i-look_back:i, :])
#         y.append(data[i+1, 8])
#     return np.array(X), np.array(y)

# train_data, test_data=train_test_split(scaled_data, test_size=0.4, shuffle=False)
# val_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=False)

# X_train, Y_train=create_dataset(train_data, look_back)
# X_test, Y_test=create_dataset(test_data, look_back)
# X_val, Y_val=create_dataset(val_data, look_back)

# model = Sequential()
# model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(LSTM(32))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear'))

# optimizer=Adam(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss='mean_squared_error')

# early_stopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_data=(X_val, Y_val), callbacks=[early_stopping])

# predicted_prices_scaled=model.predict(X_test)
# print(len(X_test))

# print(len(predicted_prices_scaled))

# print((predicted_prices_scaled))

# close_min=scaling_factors_df.iloc[8,1]
# close_max=scaling_factors_df.iloc[8,2]

# predicted_prices_unscaled=predicted_prices_scaled*(close_max-close_min)+close_min

# print(len(predicted_prices_unscaled))
# True_closing_values=Main_df.iloc[:,9]
# na_data, True_closes=train_test_split(True_closing_values, test_size=0.4, shuffle=False)
# val_data, True_closes = train_test_split(True_closes, test_size=0.5, shuffle=False)
# True_closes=True_closes.iloc[9:]
# print(len(True_closes))

import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(len(predicted_prices_unscaled))

# plt.plot(x, predicted_prices_unscaled, label='Predicted Closes', color='blue') 
# plt.plot(x, True_closes, label='True Closing Values', color='red')

# # Add labels, a legend, and a title
# plt.xlabel('Trading Days')
# plt.ylabel('SPY points')
# plt.legend()
# plt.title('Predicted vs Actual Closing levels (unscaled)')

# # Show the plot
# plt.show()

# x = np.arange(len(Y_test))


# # Plot both sets of data on the same plot
# plt.plot(x, Y_test, label='True Closing Levels', color='blue')  # You can customize the label and color
# plt.plot(x, predicted_prices_scaled, label='Predicted Closing Levels', color='red')

# # Add labels, a legend, and a title
# plt.xlabel('Trading Days')
# plt.ylabel('scaled down 0-1 SPY close')
# plt.legend()
# plt.title('Predicted close vs Actual Close (scaled down)')

# # Show the plot
# plt.show()


######## Next model
spy_data = spy_data[['Open', 'High', 'Low', 'Close']]

def create_dataset(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)-1):
        X.append(data[i-look_back:i, :])
        y.append(data[i, 3])
    return np.array(X), np.array(y)

feature_scaler2=MinMaxScaler()
scaled_data2=feature_scaler2.fit_transform(spy_data)
min_values2 = feature_scaler2.data_min_
max_values2 = feature_scaler2.data_max_


scaling_factors_df2 = pd.DataFrame({'Feature': spy_data.columns,
                                   'Min Value': min_values2,
                                   'Max Value': max_values2})

train_data2, test_data2=train_test_split(scaled_data2, test_size=0.4, shuffle=False)
val_data2, test_data2 = train_test_split(test_data2, test_size=0.5, shuffle=False)

X_train2, Y_train2=create_dataset(train_data2, look_back)
X_test2, Y_test2=create_dataset(test_data2, look_back)
X_val2, Y_val2=create_dataset(val_data2, look_back)

model2 = Sequential()
model2.add(InputLayer(input_shape=(X_train2.shape[1], X_train2.shape[2])))
model2.add(LSTM(32))
#model2.add(LSTM(32))
model2.add(Dense(8, activation='relu'))
model2.add(Dense(1, activation='linear'))

optimizer=Adam(learning_rate=0.001)
model2.compile(optimizer=optimizer, loss='mean_squared_error')

early_stopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

model2.fit(X_train2, Y_train2, epochs=50, batch_size=8, validation_data=(X_val2, Y_val2), callbacks=[early_stopping])

predicted_prices_scaled2=model2.predict(X_test2)

x = np.arange(len(predicted_prices_scaled2))

plt.plot(x, predicted_prices_scaled2, label='Predicted Closes', color='blue') 
plt.plot(x, Y_test2, label='True Closing Values', color='red')


plt.xlabel('Trading Days')
plt.ylabel('SPY points')
plt.legend()
plt.title('JUST SPY DATA - Predicted vs Actual Closing levels (unscaled)')

plt.show()

