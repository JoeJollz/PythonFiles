# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:13:13 2023

@author: jrjol
"""
""" 
This code is related to Joe's Markov Chain paper: MC_P5.
A base code for calculating state transition probability matricies.


We have 8 states here. G1,G2,G3,G4. D1,D2,D3,D4. (Please refer to the pdf MC_P5 for 
a better mathematical definition of the states. But, in summary, the states capture cumulative market trends.
When a series of green candles (G sate) is interutped by a red candle, the trend is broke and a new trend begins,
hence a trend of canldes continiously/consecutively appearing after the previous candle. 
There various catagories of G1, G2, G3, G4 simply represent the significance of that given trend,
same applies to the D states. As the state definition is based on cumulative trends, as an example take a G2 state,
a G2 state could become a G2 state at the next point in time or a G3 or G4 (if the uptrend comes more significant), but
as mentioned this uptrend can be interputed by a red candle, resulting in a D state, but note, a G2 state cannot become a G1
state at the next point in time.

The trends are calculated in the first loop through our dataset. Then the second loop is a counter,
counting the state transtions for a given state to a new state. 

Understanding the results: This is a simple code just for checking the success of calculations and programming.
I will take a few of the resulting variables, and you should be able to visualise the probability state transition matrix.

Pr_D1_D1 corresponds to the probability of a state transition from D1 (current state) to D1 (state at the next point in time).
In this case, Pr_D1_D1 = 0.2613 (26.13%).

Pr_D1_G2 corresponds to the probability of a state transition from D1 (current state) to a G2 (state at the next point in time).
In this case, Pr_D1_G2 = 0.0667 (6.67%).

Sum_D3_Gs corresponds to the probability of a state transitioning from a D3 state (current state) to any of the G states (at the 
next point in time). 
In this case, Sum_D3_Gs = 0.5846 (58.46%).

"""


import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="max_sharpe transforms the optimization problem")
from sklearn.model_selection import train_test_split

symbol = "SPY"

spy_data = yf.download(symbol, start="2015-01-01", end="2023-05-31")
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

spy_data, test_data = train_test_split(spy_data, test_size=0.3)
        
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

# Setting counters equal to zero.

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

open_position=0
ROIs=[]
lose_trades=0
win_trades=0
trades=0
for I in range(1,len(List_y_t)-1):
    Stand_Dev_prev=spy_data.iloc[I+look_back,2]
    Stand_Dev_curr=spy_data.iloc[I+look_back,2]
    
    ## We open a long position here based off reading our state transition probabilities. 
    ## If the current state is a D3 state, we open a long position.
    
    
    if -2*Stand_Dev_prev>List_y_t[I-1]>-3*Stand_Dev_prev and open_position==0: 
        buy_long=spy_data.iloc[I+look_back+1,0]
        open_position=1
    if open_position==1:
        sell_long=spy_data.iloc[I+look_back+1,1]
        ROI=(sell_long-buy_long)/buy_long
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
        