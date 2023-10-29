# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:55:04 2023

@author: jrjol
"""
RSI_period = 14
MA_window = 14
look_back= 14
#features=['Close', 'Open', 'High', 'Low', 'Volume', 'SMA', 'EMA_10', 'EMA_30', 'RSI', 'STD_10', 'PC' ]
features=['Close', 'Open', 'High', 'Low', 'Volume' ]
#features=['Close', 'SMA', 'EMA_10', 'EMA_30', 'RSI', 'STD_10', 'PC' ]

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

symbol = "SPY"

original_spy_data = yf.download(symbol, start="2015-01-01", end="2023-05-31")
original_spy_data.reset_index(inplace=True)
original_spy_data.set_index('Date', inplace=True)

def add_rsi_column(original_df, column="Close", new_column_name="RSI"):
    rsi_p = RSI_period
    
    RSI2_df = pd.DataFrame()
    RSI2_df["Change"]=original_df[column].diff()
    RSI2_df["Gain"]=np.where(RSI2_df["Change"] > 0, RSI2_df["Change"], 0 )
    RSI2_df["Loss"]=np.where(RSI2_df["Change"] < 0, abs(RSI2_df["Change"]), 0)

    avg_gains = [np.nan for i in range(rsi_p - 1)]  # Keep sizes consistent
    avg_losses = [np.nan for i in range(rsi_p - 1)]
    avg_gains.append(RSI2_df["Gain"][:14].mean())
    avg_losses.append(RSI2_df["Loss"][:14].mean())

    for i in range(rsi_p, len(RSI2_df)):
        avg_gains.append( (avg_gains[i-1] * (rsi_p - 1) 
                          + RSI2_df["Gain"][i])
                          / rsi_p)
        avg_losses.append( (avg_losses[i-1]* (rsi_p - 1) 
                          + RSI2_df["Loss"][i])
                          / rsi_p)
    RSI2_df["Average Gain"] = avg_gains
    RSI2_df["Average Loss"] = avg_losses
    RSI2_df["RS"]=RSI2_df["Average Gain"] / RSI2_df["Average Loss"]
    RSI2_df["RSI"]=100-(100/(1+RSI2_df["RS"]))
    
    original_df[new_column_name] = RSI2_df["RSI"]
    
def add_ma_columns(original_df, column="Close"):
    original_df["SMA"]=original_df[column].rolling(window = MA_window).mean()
    original_df["EMA_10"]=original_df[column].ewm(span=10).mean()
    original_df["EMA_30"]=original_df[column].ewm(span=30).mean()
    
def add_other_stats(original_df, column="Close"):
    original_df['PC']=spy_data['Close'].pct_change()
    original_df['STD_10']=original_df[column].rolling(window=10).std()
    
spy_data = original_spy_data.copy()  # Keeping original for reference

## Add features
add_rsi_column(spy_data)
add_ma_columns(spy_data)
add_other_stats(spy_data)

## Remove NaNs
spy_data.dropna(inplace=True)

## Remove unused columns
spy_data=spy_data[features]

## Keep prescaled
spy_data_pre_scale=spy_data.copy()

## Utility method for creating X/Y split
def create_dataset(spy_data, look_back):
    X, y = [], []
    for i in range(look_back, len(spy_data)-1):
        X.append(spy_data[i-look_back:i, :])
        y.append(spy_data[i, 0])    # i here, as [...:i] from previous line is upto but not including i
    return np.array(X), np.array(y)

feature_scaler=MinMaxScaler()
scaled_data=feature_scaler.fit_transform(spy_data)

target_scaler=MinMaxScaler()
target_scaler.fit(spy_data['Close'].values.reshape(-1, 1))

train_data, test_data=train_test_split(scaled_data, test_size=0.2, shuffle=False)
val_data, test_data=train_test_split(test_data, test_size=0.75, shuffle=False)

X_train, y_train=create_dataset(train_data, look_back)
X_test, y_test=create_dataset(test_data, look_back)
X_val, y_val=create_dataset(val_data, look_back)
loss_functions = [
    'mean_squared_error',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'mean_squared_logarithmic_error',
    'binary_crossentropy',
    'categorical_crossentropy',
    'sparse_categorical_crossentropy',
    'kullback_leibler_divergence',
    'hinge',
    'squared_hinge',
    'categorical_hinge',
    'huber_loss',
    'poisson',
    'cosine_proximity'
]


def ModelBuilder(X_train, y_train, X_val, y_val, loss_funct):
    model = Sequential()
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(LSTM(32))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    optimizer=Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_funct)
    early_stopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
    
    n=20
    predicted_prices_TRAIN = model.predict(X_train) 
    plt.scatter(range(n),predicted_prices_TRAIN[:n], color="blue", label="Predicted")
    plt.scatter(range(n),y_train[:n], color="orange", label="Actual")
    plt.plot(predicted_prices_TRAIN[:n], color="blue")
    plt.plot(y_train[:n], color="orange")
    for i in range(n):
        plt.plot([i,i],[predicted_prices_TRAIN[i][0], y_train[i]], color="black")
    plt.legend()
    plt.show()
    
    plt.plot(predicted_prices_TRAIN[1400:1500], color="blue", label="PREDICTED")
    plt.plot(y_train[1400:1500], color="orange", label="ACTUAL")
    plt.legend()
    plt.title(f"Predicted vs Actual on training set (lag corrected) - Loss Function: {loss_funct}")
    plt.show()
        
    (abs(predicted_prices_TRAIN.reshape(1667,) - y_train)).mean()
    
    plt.plot(predicted_prices_TRAIN.reshape(1667,) - y_train)
    plt.title("Error chart for prediction on training data")
    plt.show()
    
    
    
    
    predicted_prices_scaled=model.predict(X_test)
    predicted_prices=target_scaler.inverse_transform(predicted_prices_scaled)
    actual_prices=target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.plot(predicted_prices_scaled[100:200], color="blue", label="PREDICTED")
    plt.plot(y_test[100:200], color="orange", label="ACTUAL")
    plt.legend()
    plt.title(f"Predicted vs Actual on test set - Loss Function: {loss_funct}")
    plt.show()

for loss_funct in loss_functions:
    ModelBuilder(X_train, y_train, X_val, y_val, loss_funct)