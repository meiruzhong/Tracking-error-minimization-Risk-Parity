#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:04:27 2022

@author: zhongmeiru
"""
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import risk_opt_2Student as riskopt 
import random

#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
#import yfinance as yf
#yf.pdr_override() 
import datetime 
from scipy import optimize
import warnings
warnings.filterwarnings("ignore")


def tracking_error(wts_active,cov):
    TE = np.sqrt(np.transpose(wts_active)@cov@wts_active)
    return TE

def ewma_cov(rets, lamda,window): 
    T, n = rets.shape
    #ret_mat = rets.as_matrix()
    ret_mat = rets.values
    EWMA = np.zeros((T+1,n,n))# corection changed from T to T+1
    S = np.cov(ret_mat.T)  
    EWMA[0,:] = S
    for i in range(1, T+1) :# corection changed from T to T+1
        S = lamda * S  + (1-lamda) * np.matmul(ret_mat[i-1,:].reshape((-1,1)), 
                      ret_mat[i-1,:].reshape((1,-1)))
        EWMA[i,:] = S

    EWMA[:window-1] = np.nan
    return(EWMA)

# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  #return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])) # old
  # new - force it not to sort by the ticker alphabetically
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'],sort=False))

# function to get the return data calculated from price data 
# retrived from yahoo finance 
def getReturns(tickers, start_dt, end_dt, freq='monthly'): 
    px_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    px = px_data[['Adj Close']].reset_index().pivot(index='Date', 
                           columns='Ticker', values='Adj Close')
    if (freq=='monthly'):
        px = px.resample('M').last()
        
    # Calculate the daily/monthly percentage change
    ret = px.pct_change().dropna()
    
    ret.columns = tickers
    return(ret)

Price_AllStock = pd.read_csv('IndexPrice_XLV_2014.csv',
                  index_col = 0)
IndexData = pd.read_csv('equity_index_2014.csv',
                 index_col = 0)

ret_AllStock = Price_AllStock.pct_change().dropna()
ret_Idx = IndexData['Adj Close'].pct_change().dropna().to_frame('Return')
    
train_stock = ret_AllStock.loc[:'7/30/2020',:]
train_index = ret_Idx.loc[:'7/30/2020',:]

test_stock = ret_AllStock.loc['7/30/2020':,:]
test_index = ret_Idx.loc['7/30/2020':,:]

TickerNWeights = pd.read_csv('equity_holdings_2014.csv')
print(TickerNWeights.columns)
Ticker_AllStock = TickerNWeights['Symbol']
Ticker_AllStock = list(Ticker_AllStock)
wts_AllStock = TickerNWeights['Weight'] 

lamda = 0.94
n_period =20
# vol of the assets 
vols = train_stock.std()
rets_mean = train_stock.mean()
# demean the returns
train_data = train_stock - rets_mean
    
#training_period_cov
# var_ewma calculation of the covraiance using the function from module risk_opt.py
var_ewma = ewma_cov(train_data, lamda,n_period)
var_ewma = var_ewma

#var_ewma_annual = var_ewma*252 #Annualize
# take only the covariance matrix for the last date, which is the forecast for next time period
cov_end = var_ewma[-1,:] 
#
cov_end_annual = cov_end*252#Annualize
std_end_annual = np.sqrt(np.diag(cov_end))*np.sqrt(252)
    
# Calculate conditional mean and covariance of the factor returns, given the regime flags
# Two regimes: Risk-on (good for equity) or RiskOff (bad for equity)
df_Regime = pd.read_excel('Flag_2014.xlsx', sheet_name='Flag',
                    header=0, index_col = 0)

print(df_Regime.columns)
df_Regime.index = ret_AllStock.index
Name_Factors = ret_AllStock.columns
#training period
df_Regime = df_Regime.loc[:'7/30/2020',:]

# merging/joining dataframe
frames = [df_Regime, train_stock]# this creats a list of two frames
# merge the two frames by simply combing them
df_merged = pd.concat(frames) # default axis is 0, which means merge the columns, keep all the rows
df_merged_C = pd.concat(frames, axis=1, join='inner') # merge only the common rows

# Calculate conditional expected return and risk
a1 = df_merged_C['RiskOn Flag']
df_merged_C_RiskOn = df_merged_C[a1]
del df_merged_C_RiskOn['RiskOn Flag']#take out the flag

df_merged_C_RiskOff = df_merged_C[~df_merged_C['RiskOn Flag']]
del df_merged_C_RiskOff['RiskOn Flag']#take out the flag

# Calculate mean, covarinace and correlation daily
mean_ret_RiskOn = df_merged_C_RiskOn.mean()
cov_ret_RiskOn = df_merged_C_RiskOn.cov()
mean_ret_RiskOff = df_merged_C_RiskOff.mean()
cov_ret_RiskOff = df_merged_C_RiskOff.cov()
cov_ret_RiskOn = np.array(cov_ret_RiskOn)

# create dataframes to be exported to csv
mean_data = np.vstack((mean_ret_RiskOn, mean_ret_RiskOff))
df_mean = pd.DataFrame(mean_data, columns=Name_Factors, index=np.transpose(['RiskOn','RiskOff']))
df_cov_RiskOn = pd.DataFrame(cov_ret_RiskOn, columns=Name_Factors, index=np.transpose(Name_Factors))
df_cov_RiskOff = pd.DataFrame(cov_ret_RiskOff, columns=Name_Factors, index=np.transpose(Name_Factors))

#df_mean.to_csv('return.csv')
#df_cov_RiskOn.to_csv('risk-on.csv')
#df_cov_RiskOff.to_csv('risk-off.csv')

cov_riskon_annual = df_cov_RiskOn*252 #Annualize
cov_riskoff_annual = df_cov_RiskOff*252 #Annualize
cov_riskon_annual = np.array(cov_riskon_annual)
cov_riskoff_annual = np.array(cov_riskoff_annual)

#std_riskon_annual = np.sqrt(np.diag(cov_riskon_annual))*np.sqrt(252)
#std_riskoff_annual = np.sqrt(np.diag(cov_riskoff_annual))*np.sqrt(252)

# calculate the correlation matrix
#corr = ret_AllStock.corr()

# Calulate Covariance Matrix
#tracking error for the last day of the whole period
lamda = 0.94
n_period = 20
# vol of the assets 
vols = train_stock.std()
rets_mean = train_stock.mean()
# demean the returns
de_mean = train_stock - rets_mean
# Test case - Full Replication : minize TE to zero should produce a fund with wts like those of the index
#
# define constraints
b_ = [(0.0,1.0) for i in range(len(rets_mean))]  # no shorting 
c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })   # Sum of active weights = 100%
# calling the optimization function

wts_min_trackingerror = riskopt.opt_min_te(wts_AllStock,cov_riskon_annual, b_, c_)
# calc TE achieved
wts_active1 = wts_min_trackingerror - wts_AllStock
TE_optimized_riskon = tracking_error(wts_active1,cov_ret_RiskOn)
print('\n Daily Risk-on TE during the training period = {0:.2f} bps'.format(TE_optimized_riskon*10000))
#Daily Risk-on TE during the training period = 1.56 bps

