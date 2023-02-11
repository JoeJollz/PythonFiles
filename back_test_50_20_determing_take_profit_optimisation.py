# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:01:45 2023

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

twenDMA=[0]*20
fiftDMA=[0]*50

Take_profits_level=[]
Success_Rate=[]
PnL_per_deal=[]

Various_take_profits=np.arange(1,2,0.05)

for w in np.arange(1,4,0.05):
    Revenue=0
    buy_price=0
    
    Record_of_deals=[]
    
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
                
        elif df.iloc[i,4]<buy_price*0.91 and buy_price>0:
            
            sell_price=df.iloc[i,4]
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
        
        elif df.iloc[i,4]>buy_price*w and buy_price>0:
            sell_price=df.iloc[i,4]
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
            
            
        i=i-1
    
    Average_made_from_deal_specific_take_profit=round((pd.Series(Record_of_deals)).mean(),2)                
    Success_Rate_specific_take_profit=round((Profitable_trades/(Profitable_trades+Loss_trades))*100,2)
    
    Take_profits_level.append(w)
    Success_Rate.append(Success_Rate_specific_take_profit)
    PnL_per_deal.append(Average_made_from_deal_specific_take_profit)
    
df_results=pd.DataFrame({'Take profit level set as multiplier of the buy price': Take_profits_level, 'Success rate': Success_Rate, 'PnL per deal': PnL_per_deal})


plt.plot(Take_profits_level, Success_Rate, color='red')
plt.plot(Take_profits_level, PnL_per_deal, color='green')
plt.title("Success rate and Average P/L for each Take Profit Multiplier")
plt.xlabel("Take Profit as a multiplier of buy price")
plt.ylabel("P/L ($) and Success Rate (%)")
plt.legend(["Success Rate", "Average PnL"])