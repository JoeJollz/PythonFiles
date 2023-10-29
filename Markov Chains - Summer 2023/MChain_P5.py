# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:34:30 2023

@author: jrjol
"""

""" 
This script is very similar to MChain_P4.py 

The first loop calculates the cummulative trends. Stored in the variable List_y_t (please run and view)

The second loop is our backtesting code for caculating our state transition probabilites (on a training dataset).

The third loop calculates the cummulative trends for the testing data (unseen data).

The fourth and final loop trades our idea of state transition probabilites. If the current state lands in any of the 3 following states,
we open a long position: D3, D4 and G1.

Trading logic: we open a trade at the Open of the next day (as the current day is a D3, D4 or G1 state), we then close the position at the close of the same
trading day at which we opened the trade, as that is what our state transition probability matrix tells us, keeping the position open beyond the next trading day (next state)
is not what our strategy is designed to do. 


"""


import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="max_sharpe transforms the optimization problem")
from sklearn.model_selection import train_test_split

symbol = "SPY"

spy_data = yf.download(symbol, start="2010-01-01", end="2023-07-31")
spy_data.reset_index(inplace=True)
spy_data.set_index('Date', inplace=True)

spy_data=spy_data[['Open', 'Close']]

pct_change=spy_data['Close'].pct_change()

k=1
look_back=20
spy_data['STD']=pct_change.rolling(look_back).std()

List_y_t=[]
Index_of_occurance=[]
Y=1

spy_data, test_data = train_test_split(spy_data, test_size=0.3, shuffle=False)
        
Y=1

for I in range(look_back, len(spy_data)):
    pas=0
    if pct_change.iloc[I]>0 and Y>=1:
        Y*=(1+pct_change.iloc[I])
        List_y_t.append(Y-1)
    elif pct_change.iloc[I]<0 and Y>1:
        Y=1
        Y*=(1+pct_change.iloc[I])
        List_y_t.append(Y-1)
    elif pct_change.iloc[I]<0 and Y<=1:
        Y*=(1+pct_change.iloc[I])
        List_y_t.append(Y-1)
    elif pct_change.iloc[I]>0 and Y<1:
        Index_of_occurance.append(I)
        Y=1
        Y*=(1+pct_change.iloc[I])
        List_y_t.append(Y-1)
        pas=1
# Y=1
# for I in range(look_back, len(spy_data)):
    
#     if pct_change.iloc[I]>0 and Y>=1:
#         Y*=(1+pct_change.iloc[I])
#     if pct_change.iloc[I]<0 and Y<=1:
#         Y*=(1+pct_change.iloc[I])
#     if pct_change.iloc[I]>0 and Y<1:
        
        

P_D1=0
P_D2=0
P_D3=0
P_D4=0

P_G1=0
P_G2=0
P_G3=0
P_G4=0

P_D1_D1=0
P_D1_D2=0
P_D1_D3=0
P_D1_D4=0
P_D1_G1=0
P_D1_G2=0
P_D1_G3=0
P_D1_G4=0

P_D2_D1=0
P_D2_D2=0
P_D2_D3=0
P_D2_D4=0
P_D2_G1=0
P_D2_G2=0
P_D2_G3=0
P_D2_G4=0

P_D3_D1=0
P_D3_D2=0
P_D3_D3=0
P_D3_D4=0
P_D3_G1=0
P_D3_G2=0
P_D3_G3=0
P_D3_G4=0

P_D4_D1=0
P_D4_D2=0
P_D4_D3=0
P_D4_D4=0
P_D4_G1=0
P_D4_G2=0
P_D4_G3=0
P_D4_G4=0

P_G1_D1=0
P_G1_D2=0
P_G1_D3=0
P_G1_D4=0
P_G1_G1=0
P_G1_G2=0
P_G1_G3=0
P_G1_G4=0

P_G2_D1=0
P_G2_D2=0
P_G2_D3=0
P_G2_D4=0
P_G2_G1=0
P_G2_G2=0
P_G2_G3=0
P_G2_G4=0

P_G3_D1=0
P_G3_D2=0
P_G3_D3=0
P_G3_D4=0
P_G3_G1=0
P_G3_G2=0
P_G3_G3=0
P_G3_G4=0

P_G4_D1=0
P_G4_D2=0
P_G4_D3=0
P_G4_D4=0
P_G4_G1=0
P_G4_G2=0
P_G4_G3=0
P_G4_G4=0

for I in range(1, len(List_y_t)):
    # std_row_prev=Index_of_occurance[I-1]
    # std_row_curr=Index_of_occurance[I]
    Stand_Dev_prev=spy_data.iloc[I+look_back,2]
    Stand_Dev_curr=spy_data.iloc[I+look_back,2]
    ### From D1
    if -1*Stand_Dev_prev<List_y_t[I-1]<0: ## represents row D1
        P_D1+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_D1_D1+=1
        
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_D1_D2+=1
            
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_D1_D3+=1
        
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_D1_D4+=1
        
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_D1_G1+=1
        
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_D1_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_D1_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_D1_G4+=1
    ### From D2
    if -2*Stand_Dev_prev<List_y_t[I-1]<-1*Stand_Dev_prev: ## represents row D2
        P_D2+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_D2_D1+=1
        
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_D2_D2+=1
        
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_D2_D3+=1
        
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_D2_D4+=1   
        
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_D2_G1+=1
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_D2_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_D2_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_D2_G4+=1
    ### From D3
    if -3*Stand_Dev_prev<List_y_t[I-1]<-2*Stand_Dev_prev: ## represents row D3
        P_D3+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_D3_D1+=1
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_D3_D2+=1
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_D3_D3+=1
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_D3_D4+=1
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_D3_G1+=1
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_D3_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_D3_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_D3_G4+=1
    #### From D4
    if List_y_t[I-1]<-3*Stand_Dev_prev: ## represents row D4
        P_D4+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_D4_D1+=1
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_D4_D2+=1
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_D4_D3+=1
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_D4_D4+=1
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_D4_G1+=1
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_D4_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_D4_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_D4_G4+=1
    ### From G1
    if 0<List_y_t[I-1]<1*Stand_Dev_prev: ## represents row D1
        P_G1+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_G1_D1+=1
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_G1_D2+=1
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_G1_D3+=1
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_G1_D4+=1
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_G1_G1+=1
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_G1_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_G1_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_G1_G4+=1
    ### From G2
    if 1*Stand_Dev_prev<List_y_t[I-1]<2*Stand_Dev_prev: ## represents row D1
        P_G2+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_G2_D1+=1
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_G2_D2+=1
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_G2_D3+=1
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_G2_D4+=1
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_G2_G1+=1
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_G2_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_G2_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_G2_G4+=1
    ### From G3
    if 2*Stand_Dev_prev<List_y_t[I-1]<3*Stand_Dev_prev: ## represents row D1
        P_G3+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_G3_D1+=1
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_G3_D2+=1
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_G3_D3+=1
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_G3_D4+=1
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_G3_G1+=1
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_G3_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_G3_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_G3_G4+=1
    #### From G4
    if 3*Stand_Dev_prev<List_y_t[I-1]: ## represents row D1
        P_G4+=1
        if -1*Stand_Dev_curr<List_y_t[I]<0: ## Represents transition to column D1
            P_G4_D1+=1
        if -2*Stand_Dev_curr<List_y_t[I]<-1*Stand_Dev_curr: ## Represents transition to column D2
            P_G4_D2+=1
        if -3*Stand_Dev_curr<List_y_t[I]<-2*Stand_Dev_curr: ## Represents transition to column D3
            P_G4_D3+=1
        if List_y_t[I]<-3*Stand_Dev_curr: ## Represents transition to column D4
            P_G4_D4+=1
        if 0<List_y_t[I]<1*Stand_Dev_curr: ## Represents transition to column G1
            P_G4_G1+=1
        if 1*Stand_Dev_curr<List_y_t[I]<2*Stand_Dev_curr: ## Represents transition to column G2
            P_G4_G2+=1
        if 2*Stand_Dev_curr<List_y_t[I]<3*Stand_Dev_curr: ## Represents transition to column G3
            P_G4_G3+=1
        if 3*Stand_Dev_curr<List_y_t[I]: ## Represents transition to column G4
            P_G4_G4+=1
            

Pr_D1_D1=P_D1_D1/P_D1
Pr_D1_D2=P_D1_D2/P_D1
Pr_D1_D3=P_D1_D3/P_D1
Pr_D1_D4=P_D1_D4/P_D1
Pr_D1_G1=P_D1_G1/P_D1
Pr_D1_G2=P_D1_G2/P_D1
Pr_D1_G3=P_D1_G3/P_D1
Pr_D1_G4=P_D1_G4/P_D1

Pr_D2_D1=P_D2_D1/P_D2
Pr_D2_D2=P_D2_D2/P_D2
Pr_D2_D3=P_D2_D3/P_D2
Pr_D2_D4=P_D2_D4/P_D2
Pr_D2_G1=P_D2_G1/P_D2
Pr_D2_G2=P_D2_G2/P_D2
Pr_D2_G3=P_D2_G3/P_D2
Pr_D2_G4=P_D2_G4/P_D2

Pr_D3_D1=P_D3_D1/P_D3
Pr_D3_D2=P_D3_D2/P_D3
Pr_D3_D3=P_D3_D3/P_D3
Pr_D3_D4=P_D3_D4/P_D3
Pr_D3_G1=P_D3_G1/P_D3
Pr_D3_G2=P_D3_G2/P_D3
Pr_D3_G3=P_D3_G3/P_D3
Pr_D3_G4=P_D3_G4/P_D3

Pr_D4_D1=P_D4_D1/P_D4
Pr_D4_D2=P_D4_D2/P_D4
Pr_D4_D3=P_D4_D3/P_D4
Pr_D4_D4=P_D4_D4/P_D4
Pr_D4_G1=P_D4_G1/P_D4
Pr_D4_G2=P_D4_G2/P_D4
Pr_D4_G3=P_D4_G3/P_D4
Pr_D4_G4=P_D4_G4/P_D4

Pr_G1_D1=P_G1_D1/P_G1
Pr_G1_D2=P_G1_D2/P_G1
Pr_G1_D3=P_G1_D3/P_G1
Pr_G1_D4=P_G1_D4/P_G1
Pr_G1_G1=P_G1_G1/P_G1
Pr_G1_G2=P_G1_G2/P_G1
Pr_G1_G3=P_G1_G3/P_G1
Pr_G1_G4=P_G1_G4/P_G1

Pr_G2_D1=P_G2_D1/P_G2
Pr_G2_D2=P_G2_D2/P_G2
Pr_G2_D3=P_G2_D3/P_G2
Pr_G2_D4=P_G2_D4/P_G2
Pr_G2_G1=P_G2_G1/P_G2
Pr_G2_G2=P_G2_G2/P_G2
Pr_G2_G3=P_G2_G3/P_G2
Pr_G2_G4=P_G2_G4/P_G2

Pr_G3_D1=P_G3_D1/P_G3
Pr_G3_D2=P_G3_D2/P_G3
Pr_G3_D3=P_G3_D3/P_G3
Pr_G3_D4=P_G3_D4/P_G3
Pr_G3_G1=P_G3_G1/P_G3
Pr_G3_G2=P_G3_G2/P_G3
Pr_G3_G3=P_G3_G3/P_G3
Pr_G3_G4=P_G3_G4/P_G3

Pr_G4_D1=P_G4_D1/P_G4
Pr_G4_D2=P_G4_D2/P_G4
Pr_G4_D3=P_G4_D3/P_G4
Pr_G4_D4=P_G4_D4/P_G4
Pr_G4_G1=P_G4_G1/P_G4
Pr_G4_G2=P_G4_G2/P_G4
Pr_G4_G3=P_G4_G3/P_G4
Pr_G4_G4=P_G4_G4/P_G4

Sum_D1_Gs=Pr_D1_G1+Pr_D1_G2+Pr_D1_G3+Pr_D1_G4
Sum_D1_Ds=Pr_D1_D1+Pr_D1_D2+Pr_D1_D3+Pr_D1_D4

Sum_D2_Gs=Pr_D2_G1+Pr_D2_G2+Pr_D2_G3+Pr_D2_G4
Sum_D2_Ds=Pr_D2_D1+Pr_D2_D2+Pr_D2_D3+Pr_D2_D4

Sum_D3_Gs=Pr_D3_G1+Pr_D3_G2+Pr_D3_G3+Pr_D3_G4
Sum_D3_Ds=Pr_D3_D1+Pr_D3_D2+Pr_D3_D3+Pr_D3_D4

Sum_D4_Gs=Pr_D4_G1+Pr_D4_G2+Pr_D4_G3+Pr_D4_G4
Sum_D4_Ds=Pr_D4_D1+Pr_D4_D2+Pr_D4_D3+Pr_D4_D4

Sum_G1_Gs=Pr_G1_G1+Pr_G1_G2+Pr_G1_G3+Pr_G1_G4
Sum_G1_Ds=Pr_G1_D1+Pr_G1_D2+Pr_G1_D3+Pr_G1_D4

Sum_G2_Gs=Pr_G2_G1+Pr_G2_G2+Pr_G2_G3+Pr_G2_G4
Sum_G2_Ds=Pr_G2_D1+Pr_G2_D2+Pr_G2_D3+Pr_G2_D4

Sum_G3_Gs=Pr_G3_G1+Pr_G3_G2+Pr_G3_G3+Pr_G3_G4
Sum_G3_Ds=Pr_G3_D1+Pr_G3_D2+Pr_G3_D3+Pr_G3_D4

Sum_G4_Gs=Pr_G4_G1+Pr_G4_G2+Pr_G4_G3+Pr_G4_G4
Sum_G4_Ds=Pr_G4_D1+Pr_G4_D2+Pr_G4_D3+Pr_G4_D4


Y=1

Test_y_t=[]

for I in range(0, len(test_data)):
    pas=0
    if pct_change.iloc[len(spy_data)+I]>0 and Y>=1:
        Y*=(1+pct_change.iloc[len(spy_data)+I])
        Test_y_t.append(Y-1)
    elif pct_change.iloc[len(spy_data)+I]<0 and Y>1:
        Y=1
        Y*=(1+pct_change.iloc[len(spy_data)+I])
        Test_y_t.append(Y-1)
    elif pct_change.iloc[len(spy_data)+I]<0 and Y<=1:
        Y*=(1+pct_change.iloc[len(spy_data)+I])
        Test_y_t.append(Y-1)
    elif pct_change.iloc[len(spy_data)+I]>0 and Y<1:
        Index_of_occurance.append(len(spy_data)+I)
        Y=1
        Y*=(1+pct_change.iloc[len(spy_data)+I])
        Test_y_t.append(Y-1)
        pas=1

def Max_equity_drawdown_calc(Account_Balance_Over_Time):
    max_val_lookforward=max(Account_Balance_Over_Time)
    max_index_lookforward=Account_Balance_Over_Time.index(max_val_lookforward)
    min_val_lookforward=min(Account_Balance_Over_Time[max_index_lookforward+1:])

    min_val_lookbackward=min(Account_Balance_Over_Time)
    min_index_lookbackward=Account_Balance_Over_Time.index(min_val_lookbackward)
    max_val_lookbackward=max(Account_Balance_Over_Time[:min_index_lookbackward-1])

    lookforward_equity_drawdown=(max_val_lookforward-min_val_lookforward)/max_val_lookforward
    lookbackward_equity_drawdown=(max_val_lookbackward-min_val_lookbackward)/max_val_lookbackward

    equity_drawdown=round(max(lookforward_equity_drawdown, lookbackward_equity_drawdown)*100,1)
    return equity_drawdown

open_position=0
Account_Bal=100
Account_Balance_Over_Time=[]
ROIs=[]
lose_trades=0
win_trades=0
trades=0

for I in range(1,len(Test_y_t)-1):
    Stand_Dev_prev=test_data.iloc[I-1,2]
    Stand_Dev_curr=test_data.iloc[I,2]
    
    if -2*Stand_Dev_prev>Test_y_t[I-1]>-3*Stand_Dev_prev and open_position==0: ## represents transition from D3
        buy_long=spy_data.iloc[I,0]
        open_position=1
    elif -3*Stand_Dev_prev>Test_y_t[I-1] and open_position==0: ## represents row D4
        buy_long=spy_data.iloc[I,0]
        open_position=1
    elif 0<Test_y_t[I-1]<1*Stand_Dev_prev and open_position==0:
        buy_long=spy_data.iloc[I,0]
        open_position=1
    if  open_position==1:
        sell_long=spy_data.iloc[I,1]
        ROI=(sell_long-buy_long)/buy_long
        Account_Bal*=(1+ROI)
        Account_Balance_Over_Time.append(Account_Bal)
        # if ROI<-0.01:
        #     ROI=-0.01
        ROIs.append(ROI)
        open_position=0
        if ROI>0:
            win_trades+=1
            trades+=1
        else:
            lose_trades+=1
            trades+=1

P_winners=win_trades/trades*100
Ave=sum(ROIs)/len(ROIs)
equity_drawdown=Max_equity_drawdown_calc(Account_Balance_Over_Time)
Single_trade_equity_drawdown=min(ROIs)
        