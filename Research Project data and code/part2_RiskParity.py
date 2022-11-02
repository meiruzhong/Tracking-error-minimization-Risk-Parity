# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:52:37 2022

@author: Angie&Meiru 
"""




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 
import datetime 
from collections import OrderedDict
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")


def cov_ewma(ret_assets, lamda = 0.94):
    ret_mat = ret_assets.values
    T = len(ret_assets)
    coeff = np.zeros((T,1))
    S = ret_assets.cov()
    for i in range(1, T):
#        S = lamda * S  + (1-lamda)*np.matmul(ret_mat[i-1,:].reshape((-1,1)), 
#                          ret_mat[i-1,:].reshape((1,-1)))
        S = lamda * S  + (1-lamda)* (ret_mat[i-1,:].reshape((-1,1)) @ ret_mat[i-1,:].reshape((1,-1)) )
        
        coeff[i] = (1-lamda)*lamda**(i)
    return S/np.sum(coeff)

    
# risk budgeting approach optimisation object function
def obj_fun(W, cov_assets, risk_budget):
    var_p = np.dot(W.transpose(), np.dot(cov_assets, W))
    sigma_p = np.sqrt(var_p)
    risk_contribution = W*np.dot(cov_assets, W)/sigma_p
    risk_contribution_percent = risk_contribution/sigma_p
    return np.sum((risk_contribution_percent-risk_budget)**2)


# calculate risk budgeting portfolio weight give risk budget
def riskparity_opt(ret_assets, risk_budget, lamda, method='ewma',Wts_min=0.0, leverage=False):
    # number of assets
    num_assets = ret_assets.shape[1]
    # covariance matrix of asset returns
    if method=='ewma':
        cov_assets = cov_ewma(ret_assets, lamda)
    elif method=='ma':
        cov_assets = ret_assets.cov()
    else:
        cov_assets = cov_ewma(ret_assets, lamda)        
    
    # initial weights
    w0 = 1.0 * np.ones((num_assets, 1)) / num_assets
    # constraints
    #cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    if leverage == True:
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-2. }, # Sum of weights = 200%
              {'type':'ineq', 'fun': lambda W: W-Wts_min}) # weights greater than min wts
    else:
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. }, # Sum of weights = 100%
              {'type':'ineq', 'fun': lambda W: W-Wts_min}) # weights greater than min wts
    # portfolio optimisation
    return minimize(obj_fun, w0, args=(cov_assets, risk_budget), method='SLSQP', constraints=c_)


# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))


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
    

#%% get historical stock price data
if __name__ == "__main__":
    
    Flag_downloadData = True
    # define the time period 
    start_dt = datetime.datetime(2014, 7, 31)
    end_dt = datetime.datetime(2022, 4, 19)
    
    if Flag_downloadData:
    #    price_SPX = pdr.get_data_yahoo('SPY', start=start_dt, end=end_dt)
    #    price_AGG = pdr.get_data_yahoo('AGG', start=start_dt, end=end_dt)
        #
        Ticker_AllAsset = ['XLV', 'TLT','GLD','XLC','VCIT']
        stock_data = getDataBatch(Ticker_AllAsset, start_dt, end_dt)
        # Isolate the `Adj Close` values and transform the DataFrame
        price_AllAsset = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
        # Merge equity data with bond data
        # merging/joining dataframe
    #    frames = [price_SPX['Adj Close'], price_AGG['Adj Close']]# this creats a list of two frames
    #    price_AllAsset1 = pd.concat(frames, axis=1, join='inner') # merge only the common rows
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('PortfolioAsset.xlsx', engine='xlsxwriter')
        price_AllAsset.to_excel(writer, sheet_name='Price',startrow=0, startcol=0, header=True, index=True)
    else:
        price_AllAsset = pd.read_excel('PortfolioAsset.xlsx', sheet_name='Price',
                        header=0, index_col = 0)
    #%%
    # 2. Calculate ARP excess returns
    ret_assets = price_AllAsset.pct_change().dropna()
    ret_assets_demean = ret_assets - ret_assets.mean()
    num_assets = ret_assets.shape[1]
    
    lamda = 0.94
    SS = cov_ewma(ret_assets_demean, lamda)
    SS1 = cov_ewma(ret_assets, lamda)
    # Construct risk parity portfolio
    # portfolio dates - this defines the first date of portfolio construction
    datestr = ret_assets.index[ret_assets.index >= '2014-07-31']
    # previous month
    mth_previous = datestr[0]
    # initialise portfolio weights matrix
    wts = pd.DataFrame(index=datestr, columns=ret_assets.columns)
    # initialise portfolio return matrix
    ret_riskParity = pd.DataFrame(index=datestr, columns=['Risk Parity'])
    # how many rolling calendar days to use for covariance calculation
    window = 90
    Wts_min = 0.1
    risk_budget = 1.0/num_assets*np.ones([1,num_assets]) #risk-party
    #risk_budget = [0.7, 0.3]
    leverage = True
    varmodel = 'ewma'
    
    
    for t in datestr:
        # construct risk budgeting portfolio and re-balance on monthly basis
        if t.month==mth_previous:
            # keep the same portfolio weights within the month
            wts.loc[t] = wts.iloc[wts.index.get_loc(t)-1]
        else:
            # update the value of the previous month 
            mth_previous = t.month
            # re-balance the portfolio at the start of the month
            
            t_begin = t - timedelta(days=window)
            ret_used = ret_assets.loc[t_begin:t,:]
            wts.loc[t] = riskparity_opt(ret_used, risk_budget, lamda, varmodel, Wts_min, leverage).x
        # calculate risk budgeting portfolio returns
        ret_riskParity.loc[t] = np.sum(wts.loc[t] * ret_assets.loc[t])
        
    # Due to precision issue, wts could be a tiny negative number instead of zero, make them zero
    wts[wts<0]=0.0
    # Construct equal weighted portfolio
    ret_equalwted = pd.DataFrame(np.sum(1.0*ret_assets[ret_assets.index>=datestr[0]]/num_assets, axis=1), columns=['Equal Weighted'])
    # Construct 60/40 weighted portfolio
    #ret_equalwted = pd.DataFrame(np.sum(1.0*ret_assets[ret_assets.index>=datestr[0]]/num_assets, axis=1), columns=['Equal Weighted'])
    
    #%%
    # Calculate performance stats
    ret_cumu_assets = (ret_assets + 1).cumprod()
    ret_cumu_riskP = (ret_riskParity + 1).cumprod()
    ret_cumu_equalwt = (ret_equalwted + 1).cumprod()
    
    ret_annual_assets = ret_cumu_assets.iloc[-1]**(250/len(ret_cumu_assets))-1
    std_annual_assets = ret_assets.std()*np.sqrt(250)
    sharpe_ratio_assets = ret_annual_assets/std_annual_assets
    
    ret_annual_riskP = ret_cumu_riskP.iloc[-1]**(250/len(ret_cumu_riskP))-1
    std_annual_riskP = ret_riskParity.std()*np.sqrt(250)
    sharpe_ratio_riskP = ret_annual_riskP/std_annual_riskP
    
    ret_annual_equalwt = ret_cumu_equalwt.iloc[-1]**(250/len(ret_cumu_equalwt))-1
    std_annual_equalwt = ret_equalwted.std()*np.sqrt(250)
    sharpe_ratio_equalwt = ret_annual_equalwt/std_annual_equalwt
    
    #sharpe_table = [sharpe_ratio_riskP, sharpe_ratio_equalwt]
    sharpe_table = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP.values),
                     ('equal_wted', sharpe_ratio_equalwt.values),
                     )))

    sharpe_table1 = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP.values),
                     ('XLV', sharpe_ratio_assets['XLV']),
                     ('TLT', sharpe_ratio_assets['TLT']),
                     ('GLD', sharpe_ratio_assets['GLD']),
                     ('XLC',sharpe_ratio_assets['XLC']),
                     ('VCIT',sharpe_ratio_assets['VCIT']),
                     )))
    print('sharpe ratio of different strategies:\n',sharpe_table)
    print('\nsharpe ratio of strategies vs assets:\n',sharpe_table1)
    #%%
    # compare the portfolio cumulative returns
    figure_count = 1
    plt.figure(figure_count)
    figure_count = figure_count+1
    pd.concat([ret_cumu_riskP, ret_cumu_equalwt], axis=1).plot()
    plt.ylabel('Cumulative Return')
    plt.show()
    
    # compare the portfolio cumulative returns vs. asset returns
    plt.figure(figure_count)
    figure_count = figure_count+1
    pd.concat([ret_cumu_riskP, ret_cumu_assets], axis=1).plot()
    plt.ylabel('Cumulative Return')
    plt.show()
    
    # plot the historical weights of the assets
    # area plot showing the weights
    plt.figure(figure_count)
    figure_count = figure_count + 1
    wts.plot.area()
    plt.ylabel('asset weights')
    
    
    # hierarchical
import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp

# Date range
start = '2014-07-31'
end = '2022-4-19'

# Tickers of assets
tickers =  ['XLV', 'TLT','GLD','VCIT','XLC','LQD','SPSB','SHY','JNK','BNDX']

# Downloading the data
data = yf.download(tickers, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = tickers
assets = data.pct_change().dropna()

Y = assets

# Creating the Portfolio Object
port = rp.Portfolio(returns=Y)

# To display dataframes values in percentage format
pd.options.display.float_format = '{:.4%}'.format

# Choose the risk measure
rm = 'MSV'  # Semi Standard Deviation

# Estimate inputs of the model (historical estimates)
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate the portfolio that maximizes the risk adjusted return ratio
w1 = port.optimization(model='Classic', rm=rm, obj='Sharpe', rf=0.0, l=0, hist=True)

# Estimate points in the efficient frontier mean - semi standard deviation
ws = port.efficient_frontier(model='Classic', rm=rm, points=20, rf=0, hist=True)

# Estimate the risk parity portfolio for semi standard deviation
w2 = port.rp_optimization(model='Classic', rm=rm, rf=0, b=None, hist=True)

# dendgrame
ax = rp.plot_dendrogram(returns=Y, linkage='ward', k=None, max_k=10,leaf_order=True, ax=None)


#cluster map
ax = rp.plot_clusters(returns=Y, linkage='ward', k=None, max_k=10,
                      leaf_order=True, dendrogram=True, ax=None)


# Max Risk Adjusted Return Portfolio
label = 'Max Risk Adjusted Return Portfolio'
mu = port.mu
cov = port.cov
returns = port.returns

ax = rp.plot_frontier(w_frontier=ws, mu=mu, cov=cov, returns=returns,
                       rm=rm, rf=0, alpha=0.05, cmap='viridis', w=w1,
                       label=label, marker='*', s=16, c='r',
                       height=6, width=10, t_factor=252, ax=None)

#pie chart
ax = rp.plot_pie(w=w2, title='Portafolio', height=6, width=10,
                 cmap="tab20", ax=None)