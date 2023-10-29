# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 23:33:25 2023

@author: jrjol
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="max_sharpe transforms the optimization problem")
import finnhub
import requests
from bs4 import BeautifulSoup
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from cvxpy import ECOS, SCS
from sklearn.model_selection import train_test_split



# All_P_winners=[]
# All_trades=[]
# All_equity_drawdowns=[]
# All_average_rets=[]
# All_Final_Account_Values=[]
# All_single_worst_trades=[]

df_Percentage_winners = pd.DataFrame()
df_number_of_trades = pd.DataFrame()
df_Maximum_eq_drawdowns_of_Port = pd.DataFrame()
df_worst_trades = pd.DataFrame()
df_Average_ROI_per_trade = pd.DataFrame()
df_Account_value = pd.DataFrame()


for look_back in range(5,35,5):
    All_P_winners=[]
    All_trades=[]
    All_equity_drawdowns=[]
    All_average_rets=[]
    All_Final_Account_Values=[]
    All_single_worst_trades=[]
    row_names=[]
    for k in np.arange(2,9,0.5):
        symbol = "SPY"
        k=round(k,2)
        spy_data = yf.download(symbol, start="2010-01-01", end="2023-07-31")
        spy_data.reset_index(inplace=True)
        spy_data.set_index('Date', inplace=True)
        
        spy_data=spy_data[['Open', 'Close']]
        
        pct_change=spy_data['Close'].pct_change()
        
        spy_data['STD']=k*pct_change.rolling(look_back).std()
        
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
                
                
        
        P_D1=1
        P_D2=1
        P_D3=1
        P_D4=1
        
        P_G1=1
        P_G2=1
        P_G3=1
        P_G4=1
        
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
            Stand_Dev_prev=spy_data.iloc[I+look_back-1,2]
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
            min_val_lookforward=min(Account_Balance_Over_Time[max_index_lookforward:])
        
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
                buy_long=test_data.iloc[I]['Open']
                open_position=1
            elif -3*Stand_Dev_prev>Test_y_t[I-1] and open_position==0: ## represents row D4
                buy_long=test_data.iloc[I,0]
                open_position=1
            elif 0<Test_y_t[I-1]<1*Stand_Dev_prev and open_position==0: #G1
                buy_long=test_data.iloc[I,0]
                open_position=1
            elif 0>Test_y_t[I-1]>-1*Stand_Dev_prev and open_position==0: # Transition from D1
                buy_long=test_data.iloc[I,0]
                open_postion=1
            if open_position==0:
                Account_Balance_Over_Time.append(Account_Bal)
            if  open_position==1:
                sell_long=test_data.iloc[I,1]
                ROI=(sell_long-buy_long)/buy_long
                if ROI<-0.02:
                    ROI=-0.02
                Account_Bal*=(1+ROI)
                Account_Balance_Over_Time.append(Account_Bal)
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
        Final_account_value=Account_Bal/100
        All_P_winners.append(P_winners)
        All_trades.append(trades)
        All_equity_drawdowns.append(-equity_drawdown)
        All_average_rets.append(Ave)
        All_Final_Account_Values.append(Final_account_value)
        All_single_worst_trades.append(Single_trade_equity_drawdown)
        

        row_names.append(f'k={k}')
        
    df_Percentage_winners[f'Lookback {look_back}'] = All_P_winners
    df_number_of_trades[f'Lookback {look_back}'] = All_trades
    df_Maximum_eq_drawdowns_of_Port[f'Lookback {look_back}'] = All_equity_drawdowns
    df_worst_trades[f'Lookback {look_back}'] = All_single_worst_trades
    df_Average_ROI_per_trade[f'Lookback {look_back}'] = All_average_rets
    df_Account_value[f'Lookback {look_back}'] = All_Final_Account_Values
 
df_Percentage_winners.index = row_names
df_number_of_trades.index = row_names
df_Maximum_eq_drawdowns_of_Port.index = row_names
df_worst_trades.index = row_names
df_Average_ROI_per_trade.index = row_names
df_Account_value.index = row_names     

S_P_A=100
Acc=[]
for I in range(1,len(test_data)):
    ROI=(test_data.iloc[I,1]-test_data.iloc[I-1,1])/test_data.iloc[I-1,1]
    S_P_A*=(1+ROI)
    Acc.append(S_P_A)
# import matplotlib.pyplot as plt
# import numpy as np

# # Get the column and row labels
# x_labels = df_Account_value.columns
# y_labels = df_Account_value.index

# # Create meshgrid for x and y indices
# x_indices, y_indices = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))

# # Get the values from the DataFrame
# z_values = df_Account_value.values

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the curve
# ax.plot_surface(x_indices, y_indices, z_values, cmap='viridis')

# # Set x and y axis labels based on column and row indices
# ax.set_xticks(np.arange(len(x_labels)))
# ax.set_xticklabels(x_labels)
# ax.set_yticks(np.arange(len(y_labels)))
# ax.set_yticklabels(y_labels)

# # Set labels for x, y, and z axes
# ax.set_zlabel('Account Value')

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Get the column and row labels
# x_labels = df_Account_value.columns
# y_labels = df_Account_value.index

# # Create meshgrid for x and y indices
# x_indices, y_indices = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))

# # Get the values from the DataFrame
# z_values = df_Account_value.values

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the curve
# ax.plot_surface(x_indices, y_indices, z_values, cmap='viridis')

# # Set x and y axis labels based on column and row indices
# ax.set_xticks(np.arange(len(x_labels)))
# ax.set_xticklabels(x_labels)

# # Customize y-axis labels (show only every Nth label)
# y_ticks_indices = np.arange(1, len(y_labels), step=2)  # Adjust step value as needed
# y_ticks_labels = [y_labels[i] for i in y_ticks_indices]
# ax.set_yticks(y_ticks_indices)
# ax.set_yticklabels(y_ticks_labels)

# # Set labels for x, y, and z axes

# ax.set_zlabel('Account Value')

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Get the column and row labels
# x_labels = df_Account_value.columns
# y_labels = df_Account_value.index

# # Create meshgrid for x and y indices
# x_indices, y_indices = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))

# # Get the values from the DataFrame
# z_values = df_Account_value.values

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the curve
# ax.plot_surface(x_indices, y_indices, z_values, cmap='viridis')

# # Set x and y axis labels based on column and row indices
# ax.set_xticks(np.arange(len(x_labels)))
# ax.set_xticklabels(x_labels)

# # Customize y-axis labels (show only every Nth label)
# y_ticks_indices = np.arange(1, len(y_labels), step=3)  # Adjust step value as needed
# y_ticks_labels = [y_labels[i] for i in y_ticks_indices]
# ax.set_yticks(y_ticks_indices)
# ax.set_yticklabels(y_ticks_labels)


# ax.set_zlabel('Account Value')

# # Set aspect ratio to "auto" for automatic scaling
# ax.set_box_aspect((np.ptp(x_indices), np.ptp(y_indices), np.ptp(z_values)))

# # Set viewing angle to fit the plot in the image
# ax.view_init(elev=30, azim=-45)  # Adjust elevation (elev) and azimuth (azim) angles

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Get the column and row labels
# x_labels = df_Account_value.columns
# y_labels = df_Account_value.index

# # Create meshgrid for x and y indices
# x_indices, y_indices = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))

# # Get the values from the DataFrame
# z_values = df_Account_value.values

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the curve
# ax.plot_surface(x_indices, y_indices, z_values, cmap='viridis')

# # Set x and y axis labels based on column and row indices
# ax.set_xticks(np.arange(len(x_labels)))
# ax.set_xticklabels(x_labels)

# # Customize y-axis labels (show only every Nth label)
# y_ticks_indices = np.arange(0, len(y_labels), step=10)  # Adjust step value as needed
# y_ticks_labels = [y_labels[i] for i in y_ticks_indices]
# ax.set_yticks(y_ticks_indices)
# ax.set_yticklabels(y_ticks_labels)

# # Set labels for x, y, and z axes
# ax.set_xlabel('X Axis (Column Index)')
# ax.set_ylabel('Y Axis (Row Index)')
# ax.set_zlabel('Account Value')

# # Set aspect ratio and viewing angle
# ax.set_box_aspect((len(x_labels), len(y_labels), np.ptp(z_values)))
# ax.view_init(elev=30, azim=-45)  # Adjust elevation (elev) and azimuth (azim) angles

# # Show the plot
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Get the column and row labels
x_labels = df_Account_value.columns
y_labels = df_Account_value.index

# Create meshgrid for x and y indices
x_indices, y_indices = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))

# Get the values from the DataFrame
z_values = df_Account_value.values

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the curve
ax.plot_surface(x_indices, y_indices, z_values, cmap='viridis')

# Set x and y axis labels based on column and row indices
ax.set_xticks(np.arange(len(x_labels)))
ax.set_xticklabels(x_labels)

# Customize y-axis labels (show only every Nth label)
y_ticks_indices = np.arange(1, len(y_labels), step=3)  # Adjust step value as needed
y_ticks_labels = [y_labels[i] for i in y_ticks_indices]
ax.set_yticks(y_ticks_indices)
ax.set_yticklabels(y_ticks_labels)

# Set labels for x, y, and z axes
ax.set_zlabel('Account Value')

# Adjust axis limits for a slight zoom out effect
ax.set_xlim(-0.5, len(x_labels) - 0.5)
ax.set_ylim(-0.5, len(y_labels) - 0.5)

# Adjust viewing angle for a better perspective
ax.view_init(elev=30, azim=-45)

# Show the plot
plt.show()


import matplotlib.pyplot as plt

# Create a list of values


# Create a plot
import matplotlib.pyplot as plt

# Data for the first line


# Plot the first line
plt.plot(Account_Balance_Over_Time, label='Markov Strat')

# Plot the second line
plt.plot(Acc, label='Buy & Hold S&P')

# Add labels, title, and legend
plt.xlabel('Trading days')
plt.ylabel('Account Value $$$')
plt.title('Account Value Over Time')
plt.legend()

# Display the plot
plt.show()

       