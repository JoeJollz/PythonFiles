# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:36:06 2023

@author: jrjol
"""

import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import numpy as np

key = 'BC3XYR45YJ9FJJTY'
ts = TimeSeries(key, output_format='pandas')
data, meta = ts.get_daily_adjusted('TSLA', outputsize='full')

df=pd.DataFrame(data)
# i=df.shape[0]-50

# indexs=data.index

twenDMA=[0]*20
fiftDMA=[0]*50

# long=1
# openposition=0
# Short=-1
# epso = 0.5
# BuyLongPrice=0


# j=0

# Record_of_deals=[]
# Record_of_dates_closing_trades=[]
Stop_losses=[]
Success_Rate=[]
PnL_per_deal=[]


# Profitable_trades=0
# Loss_trades=0

Various_stop_losses=np.arange(0.01,1,0.01)



for w in range(0,99):
    Stop_level_percentage=Various_stop_losses[w]
    
    Revenue=0
    buy_price=0
    
    Record_of_deals=[]
    Record_of_dates_closing_trades=[]
    # Stop_losses=[]
    # Success_Rate=[]
    # PnL_per_deal=[]
    
    j=0
    
    Profitable_trades=0
    Loss_trades=0
    
    i=df.shape[0]-50
    

    while i>=1:
        twenDMAdata_today=df.iloc[i:i+20,4]
        fiftDMAdata_today=df.iloc[i:i+50,4]
        twenDMAmean_today=twenDMAdata_today.mean()
        fiftDMAmean_today=fiftDMAdata_today.mean()
        
        twenDMAdata_yesterday=df.iloc[i+1:i+21,4]
        twenDMAmean_yesterday=twenDMAdata_yesterday.mean()

        fiftDMAdata_yesterday=df.iloc[i+1:i+51,4]
        fiftDMAmean_yesterday=fiftDMAdata_yesterday.mean()        
        
        #today_price=df.loc[i,4]
        
        
        
        if twenDMAmean_today>fiftDMAmean_today and twenDMAmean_yesterday<fiftDMAmean_yesterday:
            
            buy_price=df.iloc[i,4]
            
        elif twenDMAmean_today<fiftDMAmean_today and twenDMAmean_yesterday>fiftDMAmean_yesterday and buy_price>0:
            
            sell_price=df.iloc[i,4]
            j+=1
            made_from_deal=sell_price-buy_price
            Revenue=Revenue+made_from_deal
            
            Record_of_deals.append(made_from_deal)
            
            buy_price=0            
            
            # date=indexs[i]
            # Record_of_dates_closing_trades.append(date)
            
            if made_from_deal>0:
                Profitable_trades+=1
            else:
                Loss_trades+=1
                
        elif df.iloc[i,4]<buy_price*Stop_level_percentage and buy_price>0:
            
            sell_price=df.iloc[i,4]
            j+=1
            made_from_deal=sell_price-buy_price
            Revenue=Revenue+made_from_deal
            
            Record_of_deals.append(made_from_deal)
            
            buy_price=0
            
            if made_from_deal>0:
                Profitable_trades+=1
            else:
                Loss_trades+=1
            
        i=i-1
    
    Average_made_from_deal_specific_stop_loss=round((pd.Series(Record_of_deals)).mean(),2)                
    Success_Rate_specific_stop_loss=round((Profitable_trades/(Profitable_trades+Loss_trades))*100,2)
    
    Stop_losses.append(Stop_level_percentage)
    Success_Rate.append(Success_Rate_specific_stop_loss)
    PnL_per_deal.append(Average_made_from_deal_specific_stop_loss)
    
df_results=pd.DataFrame({'Stop loss used (%) of buy price': Stop_losses, 'Success rate': Success_Rate, 'PnL per deal': PnL_per_deal})


plt.plot(Stop_losses, Success_Rate, color='red')
plt.plot(Stop_losses, PnL_per_deal, color='green')
plt.title("Success rate and Average P/L for each stop loss level")
plt.xlabel("Stop loss as a % of buy price")
plt.ylabel("P/L ($) and Success Rate (%)")
plt.legend(["Success Rate", "Average PnL"])

        
               
########  Post Processing ##########

# Average_made_from_deal=(pd.Series(Record_of_deals)).mean()                
# Success_Rate=(Profitable_trades/(Profitable_trades+Loss_trades))*100
                
# df_results=pd.DataFrame({'Dates for closing positions':Record_of_dates_closing_trades, 'Profit/Loss $USD$':Record_of_deals})

# print("Success rate of this strategy is: " + str(round(Success_Rate, 2)) + "%")
# print("The average profit/loss from each trade is: $USD$"+ str(round(Average_made_from_deal, 2)))
# print(df_results)

# plt.plot(Record_of_dates_closing_trades, Record_of_deals)
# plt.xlabel("DATE")
# plt.ylabel("Profit/Loss $USD$")
