#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:21:17 2022

@author: zhongmeiru
"""


#v2 - tracking error - forecast realized - changing period from 7/21/2014 to 4/19/2022
#delete MRNA,OGN from the bechmark portfolio, because of lack of the data

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

if __name__ == "__main__":
    
    TickerNWeights = pd.read_csv('equity_holdings_2014.csv')
    print(TickerNWeights.columns)
    Ticker_AllStock = TickerNWeights['Symbol']
    Ticker_AllStock = list(Ticker_AllStock)
    wts_AllStock = TickerNWeights['Weight']           
    #Price_AllStock_DJ = TickerNWeights['Price']
    
    #%% get historical stock price data
    Flag_downloadData = False
    # define the time period 
    start_dt = datetime.datetime(2014, 7, 31)
    end_dt = datetime.datetime(2022, 4, 20)
    
    if Flag_downloadData:
        IndexData = pdr.get_data_yahoo('XLV', start=start_dt, end=end_dt)
        #
        stock_data = getDataBatch(Ticker_AllStock, start_dt, end_dt)
        # Isolate the `Adj Close` values and transform the DataFrame
        Price_AllStock = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
        Price_AllStock = Price_AllStock[list(Ticker_AllStock)]
        Price_AllStock.to_csv('IndexPrice_XLV_2014.csv')
        IndexData.to_csv('equity_index_2014.csv')
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        #writer = pd.ExcelWriter('IndexPrice_XLV.xlsx', engine='xlsxwriter')
        #Price_AllStock.to_excel(writer, sheet_name='Price',startrow=0, startcol=0, header=True, index=True)
        #Data.to_excel(writer, sheet_name='Price',startrow=0, startcol=0, header=True, index=True)
    else:
        Price_AllStock = pd.read_csv('IndexPrice_XLV_2014.csv',
                         index_col = 0)
        IndexData = pd.read_csv('equity_index_2014.csv',
                        index_col = 0)
    
    #%%
    # Returns
    ret_AllStock = Price_AllStock.pct_change().dropna()
    ret_Idx = IndexData['Adj Close'].pct_change().dropna().to_frame('Return')
    
    train_stock = ret_AllStock.loc[:'7/30/2021',:]
    train_index = ret_Idx.loc[:'7/30/2021',:]
    
    test_stock = ret_AllStock.loc['7/30/2021':,:]
    test_index = ret_Idx.loc['7/30/2021':,:]
    
    # Scale return data by a factor. It seems that the optimizer fails when the values are too close to 0
    scale = 1
    train_stock = train_stock*scale
    #
    num_periods, num_stock = train_stock.shape
    n_period = 20
    
    #%%
    # Calulate Training model's Covariance Matrix
    lamda = 0.94
    # vol of the assets 
    vols = train_stock.std()
    rets_mean = train_stock.mean()
    # demean the returns
    train_data = train_stock - rets_mean
    
    #training_period_cov
    # var_ewma calculation of the covraiance using the function from module risk_opt.py
    var_ewma = ewma_cov(train_data, lamda,n_period)
    
    #var_ewma_annual = var_ewma*252 #Annualize
    # take only the covariance matrix for the last date, which is the forecast for next time period
    cov_end = var_ewma[-1,:]
    cov_end_annual = cov_end*252#Annualize
    
    #std_end_annual = np.sqrt(np.diag(cov_end))*np.sqrt(252)
    # calculate the correlation matrix
    #corr = train_stock.corr()
    
    #%% Calulate test model's Covariance Matrix
    lamda = 0.94
    # vol of the assets 
    vols1 = test_stock.std()
    rets_mean1 = test_stock.mean()
    # demean the returns
    test_data = test_stock - rets_mean1
    
    # var_ewma calculation of the covraiance using the function from module risk_opt.py
    var_ewma1 = ewma_cov(test_data, lamda,n_period)
    #var_ewma_annual = var_ewma*252 #Annualize
    # take only the covariance matrix for the last date, which is the forecast for next time period
    cov_end1 = var_ewma1[-1,:]
    #
    cov_end_annual1 = cov_end1*252 #Annualize
    #std_end_annual1 = np.sqrt(np.diag(cov_end1))*np.sqrt(252)
    # calculate the correlation matrix
    #corr1 = test_stock.corr()
    
    #%%
    # tracking error optimization
    # Training model- Full Replication : minize TE to zero should produce a fund with wts like those of the index
    #
    # define constraints
    b_ = [(0.0,1.0) for i in range(len(rets_mean))]  # no shorting 
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })   # Sum of active weights = 100%
    # calling the optimization function
    #training model's parameters
    wts_min_trackingerror = riskopt.opt_min_te(wts_AllStock, cov_end_annual, b_, c_)
    # calc TE achieved
    wts_active1 = wts_min_trackingerror - wts_AllStock
    #training model's TE
    #forecasting period using rolling window 10 days
    TE_optimized = tracking_error(wts_active1,cov_end*n_period)
    print('\nThe EWMA daily TE for the training period = {0:.2f} bps'.format(TE_optimized*10000))
    #The EWMA daily TE for the training period = 0.32 bps
    
    #training model's forecasting daily TE
    numtest = len(var_ewma1)-1
    TE_optimized1 = np.zeros((numtest,1))
    for i in range(numtest):
        TE_optimized1[i] = tracking_error(wts_active1,cov_end)*n_period
        
    #%%   
    #test model
    #foreward-realized TE 
    # calculate the forward-realized covariance
    cov_FwdRealized = np.full((numtest,62,62), np.nan)
    # Note we only care about the forward realized std, starting from period 250, because we intend to compare 
    # them with the ones based on backward-looking ma and ema models
    for t in range(n_period, numtest-n_period+1):
        a= test_data[t : t+n_period].T
        cov_FwdRealized[t] = np.cov(a)
        del a
    
    TE_optimized2 = np.zeros((numtest,1))
    for i in range(numtest):
        TE_optimized2[i] = tracking_error(wts_active1,cov_FwdRealized[i,:])

    #
    dates = pd.Series(test_stock.index).astype(str)
    
    TE_optimized1 = TE_optimized1.flatten()
    TE_optimized1 = TE_optimized1*10000
    TE_optimized2 = TE_optimized2.flatten()
    TE_optimized2 = TE_optimized2*10000
    
    cov_annual_forward = cov_FwdRealized[n_period,:]
    TE_forward_1 = tracking_error(wts_active1,cov_annual_forward)
    TE_forward_1 = float(TE_forward_1*10000)
    TE_forward_1 = round(TE_forward_1,2)
    print('\nThe Forward-realized daily TE = {0:.2f} bps\n'.format(TE_forward_1))

#%%
#comparison between training results and test results
import matplotlib.dates as mdates
from datetime import datetime
figure_count = 1
fig2=plt.figure(figure_count, figsize=(12, 10), edgecolor='k')
figure_count = figure_count+1
a=pd.to_datetime(dates)

ax1 = plt.subplot(facecolor='w')
ax1.plot(a,TE_optimized1,'k-', linewidth=2, label = 'Forecasted TE (bps)',)
ax1.plot(a,TE_optimized2,'r-.', linewidth=1, label = 'Forward realized TE (bps)')
plt.ylim(0,10)
#ax1.legend(loc='upper center', ncol=2)
ax1.legend(loc='upper left', ncol=10,fontsize = 'x-large')

#ax1.xaxis.set_tick_params(reset=True)
#ax1.xaxis.set_major_locator(mdates.YearLocator())
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.ylabel('Tracking error', fontweight = 'bold')
 

#%%
# EWMA top 35 stoks
#use a part of stocks to duplicate the bechmark portfoilo
# Test case - use only the top market cap stocks with highest index weights
num_topwtstock_2include = 35
# NUM =30, ERROR = 60.42BPS
# NUM =33, ERROR = 49.45BPS
# NUM =35, ERROR = 43.74 BPS 

# only the top weight stocks + no shorting 
b1a_ = [(0.0,1.0) for i in range(num_topwtstock_2include)]
# exclude bottom weighted stocks
b1b_ = [(0.0,0.0000001) for i in range(num_topwtstock_2include,num_stock)]
b1_ = b1a_ + b1b_ # combining the constraints
#b1_[num_topwtstock_2include:-1] = (0.0,0.0)
c1_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })  # Sum of active weights = 100%
# Calling the optimizer
wts_min_trackingerror2 = riskopt.opt_min_te(wts_AllStock, cov_end_annual, b1_, c1_)
# calc TE achieved
wts_active2 = wts_min_trackingerror2 - wts_AllStock
TE_optimized4 = tracking_error(wts_active2,cov_end)
print('\n{0} top weighted stocks replication daily EWMA TE = {1:5.2f} bps'.format(num_topwtstock_2include, TE_optimized4*10000))

###
# looping through number of stocks and save the history of TEs
num_stock_b = 10
num_stock_e = 40
numstock_2use = range(num_stock_b,num_stock_e)
wts_active_hist = np.zeros([len(numstock_2use), num_stock])
TE_hist = np.zeros([len(numstock_2use), 1])
count = 0

for i in numstock_2use:
    # only the top weight stocks + no shorting 
    b1_c_a_ = [(0.0,1.0) for j in range(i)] 
    # exclude bottom weighted stocks
    b1_c_b_ = [(0.0,0.0000001) for j in range(i,num_stock)] 
    b1_curr_ = b1_c_a_ + b1_c_b_
    wts_min_curr = riskopt.opt_min_te(wts_AllStock, cov_end_annual, b1_curr_, c1_)
    wts_active_hist[count,:] = wts_min_curr.transpose()
    TE_optimized_c = tracking_error(wts_min_curr-wts_AllStock,cov_end_annual)
    TE_hist[count,:] = TE_optimized_c*10000# in bps
    count = count+1
    
    del b1_curr_, b1_c_a_, b1_c_b_,TE_optimized_c,wts_min_curr
#

#------plot TE as a function of number of stocks -------------
plt.figure(figure_count)
figure_count = figure_count+1
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(range(num_stock_b,num_stock_e), TE_hist, 'b')
plt.xlabel('Number of stocks in ETF', fontsize=18)
plt.ylabel('Optimized Tracking Error (bps)', fontsize=18)
plt.title('Index ETF', fontsize=18)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)

#%%%
#  Plot bars of weights
#
figure_count = 1
# ---  create plot of weights fund vs benchmark
plt.figure(figure_count)
figure_count = figure_count+1
fig, ax = plt.subplots(figsize=(18,10))
index = np.arange(len(wts_AllStock))
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, wts_AllStock, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Index Weight')
 
rects2 = plt.bar(index + bar_width, wts_min_trackingerror2, bar_width,
                 alpha=opacity,
                 color='g',
                 label='ETF fund Weight')
 
plt.xlabel('Ticker', fontsize=18)
plt.ylabel('Weights', fontsize=18)
plt.title('Regular Index ETF', fontsize=18)
plt.xticks(index + bar_width, (Ticker_AllStock), fontsize=12)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(fontsize=20)
 
plt.tight_layout()
plt.show()

#%%
#Random choice for portfolio built
np.random.seed(222)
#find weights from the training set
totalstock = list(range(test_stock.shape[1]))
selected = random.sample(totalstock,35)

set_difference = set(totalstock).symmetric_difference(set(selected))
set_difference = list(set_difference)
a = list(np.hstack((selected,set_difference)))
b = [(0,1)]*len(selected)
c = [(0.0,0.0000001)] *len(set_difference)
d = b+c

d = pd.Series(d)
d1 = d.reindex(a)
d1 = list(d1)

# only the top weight stocks + no shorting 
#b1aa_ = [(0.0,1.0) for i in range(35)]
# exclude bottom weighted stocks
#b1bb_ = [(0.0,0.0000001) for i in range(35,62)]
#b11_ = b1aa_ + b1bb_ # combining the constraints
#b1_[num_topwtstock_2include:-1] = (0.0,0.0)
c11_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })  # Sum of active weights = 100%
# Calling the optimizer
wts_min_trackingerror3 = riskopt.opt_min_te(wts_AllStock, cov_end_annual, d1, c11_)
# calc TE achieved
wts_active3 = wts_min_trackingerror3 - wts_AllStock
TE_optimized3 = tracking_error(wts_active3,cov_end)
print('\n{0} random weighted stock replication daily TE = {1:5.2f} bps'.format(len(selected), TE_optimized3*10000))


#random choice weights chart
figure_count = 1
# ---  create plot of weights fund vs benchmark
plt.figure(figure_count)
figure_count = figure_count+1
fig, ax = plt.subplots(figsize=(18,10))
index = np.arange(len(wts_AllStock))
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, wts_AllStock, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Index Weight')
 
rects2 = plt.bar(index + bar_width, wts_min_trackingerror3, bar_width,
                 alpha=opacity,
                 color='g',
                 label='ETF fund Weight')
 
plt.xlabel('Ticker', fontsize=18)
plt.ylabel('Weights', fontsize=18)
plt.title('Random Index ETF', fontsize=18)
plt.xticks(index + bar_width, (Ticker_AllStock), fontsize=12)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(fontsize=20)
 
plt.tight_layout()
plt.show()

