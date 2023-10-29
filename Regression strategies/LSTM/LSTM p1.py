# -*- coding: utf-8 -*-
## Importing some relavent files ##

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#### This is an LSTM for the SPY ###

### Note throughout the code, it certain looks messy, some coding lines have been commented out for now.
symbol = "SPY"

spy_data = yf.download(symbol, start="2015-01-01", end="2023-05-31")
spy_data.reset_index(inplace=True)
spy_data.set_index('Date', inplace=True)

##### We will calculate some technical indicators to work with the classical:
##### Open, high, low, close, vol. We are adding Simple moving averages (SMA), exp moving
##### averages (EMA), Relative Strength Index (RSI) and Std. Each of this are calculated based
##### on a certain number of historical points in time (days in this case).
N=15
RSI_lookback_period=N
### Calculating the Relative strength index ####
def RSI(spy_data, RSI_lookback_period):
    RSI_df=pd.DataFrame()
    RSI_df["change"]=spy_data.iloc[:,3].diff()
    
    RSI_df["Gain"]=np.where(RSI_df["change"] > 0, RSI_df["change"], 0 )
    RSI_df["Loss"]=np.where(RSI_df["change"] < 0, abs(RSI_df["change"]), 0)
    
    RSI_df["Average Gain"]=RSI_df.iloc[:,1].rolling(window = RSI_lookback_period).mean()
    RSI_df["Average Loss"]=RSI_df.iloc[:,2].rolling(window = RSI_lookback_period).mean()
    
    RSI_df["RS"]=RSI_df["Average Gain"] / RSI_df["Average Loss"]
    
    RSI_df["RSI"]=100-(100/(1+RSI_df["RS"]))
    
    spy_data["RSI"]=RSI_df["RSI"]
    
    return spy_data

### Calc SMA and EMA and Std
def SMA_EMA(spy_data, N):

    spy_data["SMA"]=spy_data.iloc[:,3].rolling(window = N).mean()
    spy_data["EMA_10"]=spy_data.iloc[:,3].ewm(span=10).mean()
    spy_data["EMA_30"]=spy_data.iloc[:,3].ewm(span=30).mean()
    spy_data['STD_10']=spy_data['Close'].rolling(window=10).std()

    return spy_data

spy_data=RSI(spy_data, RSI_lookback_period)
spy_data=SMA_EMA(spy_data, N)
### Calculating the percentage change can be useful, the LSTM could recognise this 
### as 2 seperate classes: negative values, positive values. We could further make a
### boolean variable of 0 and 1 to help classify positive and negative changes in the 
### index prices, which may allow the LSTM to learn better.
spy_data['Return']=spy_data['Close'].pct_change()


## Removing any rows with nan, as this may result in an error when training the NN ##
spy_data.dropna(inplace=True)
### Relocating the features, and removing adj close. ### Note, we are trying to predict the close, 
### now the close is being located in column index 0, hence, if we are trying to predict the close for
### next point in time, to extract from a DataFrame we use data.iloc[t+1,0] or an array data[t+1,0]
features=['Close', 'Open', 'High', 'Low', 'Volume', 'SMA', 'EMA_10', 'EMA_30', 'RSI', 'STD_10', 'Return' ]
spy_data=spy_data[features]
### saved the scaled data, in case we need to make comparisons.
spy_data_pre_scale=spy_data

# spy_data['Close'] = spy_data['Close'].shift(-1)

#### Scaling the data to feed into the NN ###

feature_scaler=MinMaxScaler()
scaled_data=feature_scaler.fit_transform(spy_data)

### Designing a scaler to fit the dimensions of the close column we are trying to predict. Note this includes the whole
### close data column. Training, test and val closing values.
target_scaler=MinMaxScaler()
target_scaler.fit(spy_data['Close'].values.reshape(-1, 1))

# target_scaler = MinMaxScaler()
# target_scaler.fit(spy_data[ len(spy_data)-len(test_data):len(spy_data), 0].values.reshape(-1,1))


### When the NN works, it goes of a previous number of days to make a prediction, please trial with this 
### look_back variable. I tried look_back=40 and it seemed to fit better, but no way good enough.

look_back=70

### I am assuming the LSTM NN from Tensor flow requires us to correctly input the predicted variable, if someone can check this
### so that the model doesn't automatically do mirroring. I have the input features being taken as the current point in time (and look_back)
### period. Then note, column index 0 here, represents the 'Close'. We want the feature we are trying to predict to be the close at the 
### next point in time. Therefore the close at the next point in time is: spy_data[i+1,0] (array not dataframe).
def create_dataset(spy_data, look_back):
    X, y = [], []
    for i in range(look_back, len(spy_data)-1):
        X.append(spy_data[i-look_back:i, :])
        y.append(spy_data[i+1, 0])
    return np.array(X), np.array(y)

train_data, test_data=train_test_split(scaled_data, test_size=0.2, shuffle=False)
val_data, test_data=train_test_split(test_data, test_size=0.75, shuffle=False)

X_train, y_train=create_dataset(train_data, look_back)
X_test, y_test=create_dataset(test_data, look_back)
X_val, y_val=create_dataset(val_data, look_back)
print(X_train.shape[1])
# x=spy_data[['Open', 'High', 'Low', 'Volume', 'RSI', 'SMA']]
# y=spy_data['Close']
# target_scaler = MinMaxScaler()
# target_scaler.fit(spy_data.iloc[-len(test_data):, 0].values.reshape(-1, 1))


#### The LSTM model parameters, please adjust and find a better MSE and MAE

# Build the LSTM model
# model = Sequential()
# model.add(Bidirectional(LSTM(256, return_sequences=True),input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0))
# model.add(Bidirectional(LSTM(32, return_sequences=True)))
# model.add(Dropout(0))
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
# model.add(Dropout(0))
# model.add(Bidirectional(LSTM(32, return_sequences=False)))
# model.add(Dropout(0))
# model.add(Dense(1))
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))


optimizer=Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

early_stopping=EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

### Early stopping is present through monitoring the val loss. When running a certain number of epochs, each epoch 
### may result in a better fit on the training data (lower training loss), but the val loss may begin to increase
### meaning it is beginning to overfit to the noise on the training data set, rather then capturing a general trend.
### The best model weights are saved, and applied when making predictions. Based on the model with lowest val loss.
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val), callbacks=[early_stopping])
# model.fit(X_train, y_train, epochs=20, batch_size=16)

predicted_prices_scaled=model.predict(X_test)

# predicted_prices = target_scaler.inverse_transform(predicted_prices)
# actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))

#### There may lie an issue here, with inverse scaler.
predicted_prices=target_scaler.inverse_transform(predicted_prices_scaled)
actual_prices=target_scaler.inverse_transform(y_test.reshape(-1, 1))

mse=mean_squared_error(actual_prices, predicted_prices)
mae=mean_absolute_error(actual_prices, predicted_prices)
percentage_accuracy=np.mean(np.abs(predicted_prices - actual_prices) / actual_prices) * 100

print("Predicted closing price for the next day:", predicted_prices[-1][-1])
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Percentage Accuracy:", percentage_accuracy)

### graph displaying the scaled down results ####

plt.plot(y_test, label='Actual Prices')

plt.plot(predicted_prices_scaled, label='Predicted prices scaled')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Predicted Prices vs Actual Prices (Scaled)')

plt.legend()

plt.show()
##### The code below here can be ignored for now, and is likely to change. ##### 

def trade_v2(actual_prices, predicted_prices, initial_balance=5000):
    balance = initial_balance
    position = 0
    entry_price = 0
    balance_list = [initial_balance]
    
    # Calculate the predicted percentage change
    predicted_pct_change = np.diff(predicted_prices, axis=0) / predicted_prices[:-1]

    for i in range(len(actual_prices) - 1):  # We'll stop at the second-to-last price, as we have one less percentage change value
        if position == 0:
            if predicted_pct_change[i] > 0:
                # Buy
                position = 1
                entry_price = actual_prices[i]
            elif predicted_pct_change[i] < 0:
                # Sell
                position = -1
                entry_price = actual_prices[i]
        elif position == 1:
            if predicted_pct_change[i] < 0:
                # Close the long position and sell
                balance += (actual_prices[i] - entry_price) * (balance / entry_price)
                position = 0
        elif position == -1:
            if predicted_pct_change[i] > 0:
                # Close the short position and buy
                balance += (entry_price - actual_prices[i]) * (balance / entry_price)
                position = 0

        # Update the balance based on the current position
        if position == 1:
            current_balance = balance + (actual_prices[i] - entry_price) * (balance / entry_price)
        elif position == -1:
            current_balance = balance + (entry_price - actual_prices[i]) * (balance / entry_price)
        else:
            current_balance = balance

        balance_list.append(current_balance)

    return balance_list

# def plot_strategy_vs_buy_and_hold(actual_prices, strategy_balance, initial_balance=5000):
#     buy_hold_balance = [initial_balance]
#     for i in range(len(actual_prices) - 1):
#         buy_hold_balance.append(buy_hold_balance[-1] * (actual_prices[i + 1] / actual_prices[i]))

#     plt.figure(figsize=(12, 6))
#     plt.plot(strategy_balance, label='Strategy Balance')
#     plt.plot(buy_hold_balance, label='Buy and Hold Balance')
#     plt.xlabel('Time (days)')
#     plt.ylabel('Balance')
#     plt.title('Strategy vs Buy and Hold')
#     plt.legend()
#     plt.show()


### Plot 2 graphs, one showing the unscaled predictions###
strategy_balance_v2 = trade_v2(actual_prices.flatten(), predicted_prices.flatten(), initial_balance=5000)
## plot_strategy_vs_buy_and_hold(actual_prices.flatten(), strategy_balance_v2, initial_balance=5000)
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('SPY Actual vs Predicted Prices (Unscaled)')
plt.legend()
plt.show()
