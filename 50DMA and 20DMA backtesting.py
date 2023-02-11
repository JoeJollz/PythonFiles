# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:23:54 2023

@author: jrjol
"""

import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime

key = 'BC3XYR45YJ9FJJTY'
ts = TimeSeries(key, output_format='pandas')
data, meta = ts.get_daily_adjusted('TSLA', outputsize='full')

df=pd.DataFrame(data)
i=df.shape[0]-50

indexs=data.index

twenDMA=[0]*20
fiftDMA=[0]*50

long=1
openposition=0
Short=-1
Revenue=0
epso = 0.5
BuyLongPrice=0

j=0

Record_of_deals=[]
Record_of_dates_closing_trades=[]

Profitable_trades=0
Loss_trades=0


while i>=1:
    twenDMAdata_today=df.iloc[i:i+20,4]
    fiftDMAdata_today=df.iloc[i:i+50,4]
    twenDMAmean_today=twenDMAdata_today.mean()
    fiftDMAmean_today=fiftDMAdata_today.mean()
    
    twenDMAdata_yesterday=df.iloc[i+1:i+21,4]
    twenDMAmean_yesterday=twenDMAdata_yesterday.mean()
    
    
    #today_price=df.loc[i,4]
    
    
    
    if twenDMAmean_today>fiftDMAmean_today and twenDMAmean_yesterday<fiftDMAmean_today:
        
        buy_price=df.iloc[i,4]
        
    elif twenDMAmean_today<fiftDMAmean_today and twenDMAmean_yesterday>fiftDMAmean_today:
        
        sell_price=df.iloc[i,4]
        j+=1
        made_from_deal=sell_price-buy_price
        Revenue=Revenue+made_from_deal
        
        Record_of_deals.append(made_from_deal)
        
        date=indexs[i]
        Record_of_dates_closing_trades.append(date)
        
        if made_from_deal>0:
            Profitable_trades+=1
        else:
            Loss_trades+=1
            
        
    
    
    i=i-1
           
#Post Processing

Average_made_from_deal=(pd.Series(Record_of_deals)).mean()                
Success_Rate=(Profitable_trades/(Profitable_trades+Loss_trades))*100
                
df_results=pd.DataFrame({'Dates for closing positions':Record_of_dates_closing_trades, 'Profit/Loss':Record_of_deals})

print("Success rate of this strategy is: " + str(round(Success_Rate, 2)) + "%")
print("The average profit/loss from each trade is: $USD$"+ str(round(Average_made_from_deal, 2)))

plt.plot(Record_of_dates_closing_trades, Record_of_deals)
plt.xlabel("DATE")
plt.ylabel("Profit/Loss $USD$")