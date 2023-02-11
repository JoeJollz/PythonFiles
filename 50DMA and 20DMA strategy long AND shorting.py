# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:23:10 2023

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
i=df.shape[0]-51

indexs=data.index

twenDMA=[0]*20
fiftDMA=[0]*50

long_buy_price=0
short_buy_price=0
long_sell_price=0
short_sell_price=0

Short_revenue=0
Long_revenue=0

Profitable_trades=0
Loss_trades=0

Record_of_dates_closing_short_trades=[]
Record_of_long_deals=[]
Record_of_dates_closing_long_trades=[]
Record_of_short_deals=[]
Record_of_all_deals=[]

while i>=1:
    twenDMAdata_today=df.iloc[i:i+20,4]
    fiftDMAdata_today=df.iloc[i:i+50,4]
    twenDMAmean_today=twenDMAdata_today.mean()
    fiftDMAmean_today=fiftDMAdata_today.mean()
    
    twenDMAdata_yesterday=df.iloc[i+1:i+21,4]
    twenDMAmean_yesterday=twenDMAdata_yesterday.mean()
    
    fiftDMAdata_yesterday=df.iloc[i+1:i+51,4]
    fiftDMAmean_yesterday=fiftDMAdata_yesterday.mean()
    
    if twenDMAmean_today>fiftDMAmean_today and twenDMAmean_yesterday<fiftDMAmean_yesterday:
        
        short_sell_price=df.iloc[i,4]
        
        Made_from_short=short_buy_price-short_sell_price
        Short_revenue=Short_revenue+Made_from_short 
        
        Record_of_short_deals.append(Made_from_short)
        
        Record_of_all_deals.append(Made_from_short)
        
        short_date=indexs[i]
        Record_of_dates_closing_short_trades.append(short_date)
        
        if Made_from_short>0:
            Profitable_trades+=1
        else:
            Loss_trades+=1
        
        
        long_buy_price=short_sell_price
        
        
    elif twenDMAmean_today<fiftDMAmean_today and twenDMAmean_yesterday>fiftDMAmean_yesterday:
        
        long_sell_price=df.iloc[i,4]
        
        Made_from_long=long_sell_price-long_buy_price
        Long_revenue=Long_revenue+Made_from_long
        
        Record_of_long_deals.append(Made_from_long)
        
        Record_of_all_deals.append(Made_from_long)
        
        long_date=indexs[i]
        Record_of_dates_closing_long_trades.append(long_date)
        
        if Made_from_long>0:
            Profitable_trades+=1
        else:
            Loss_trades+=1
        
        short_buy_price=long_sell_price
        
        
        
    i-=1

Total_revenue=Long_revenue+Short_revenue

Average_made_from_short_deals=(pd.Series(Record_of_short_deals)).mean()
Average_made_from_Long_deals=(pd.Series(Record_of_long_deals)).mean()

Average_of_short_and_long_deals=(pd.Series(Record_of_all_deals)).mean()

Success_Rate=(Profitable_trades/(Profitable_trades+Loss_trades))*100

df_results=pd.DataFrame({'Dates for closing Short positions':Record_of_dates_closing_short_trades, 'Profit/Loss for short positions $USD$':Record_of_short_deals, 'Dates for closing Long positions': Record_of_dates_closing_long_trades, 'Profit/Loss for long positions $USD$':Record_of_long_deals})
print(df_results)
print("Success rate of this strategy is: " +str(round(Success_Rate,2)) +"%")
print("The average PnL for short trades is: " +str(round(Average_made_from_short_deals,2)) + "($USD$)")
print("The average PnL for long trades is: " + str(round(Average_made_from_Long_deals,2)) + "($USD$")
print("The average PnL for long and short trades is: " +str(round(Average_of_short_and_long_deals,2)) + "($USD$)")