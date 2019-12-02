#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:01:03 2019

@author: Sherry
"""

""" Import modules """
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import talib
from talib.abstract import *


""" Load price data from disk """
def load_data(path, price_name=None, verbose = 0):
    path_price = path
    p_df = pd.DataFrame()
    for file in sorted(os.listdir(path_price)):
        if file.find('_adj.csv')<=0:
            continue
        ticker = file[:file.find('_')]
        if not verbose:
            print(file)
        p_df1 = pd.read_csv(path_price + '/' + file)
        if ticker !='SPY':
            if len(p_df1.ticker.unique())!=1:
                warnings.warn("More than one ticker in the file")
            if  ticker!=p_df1.ticker.unique()[0]:
                warnings.warn("Ticker does not match with the file name")
            if price_name is not None:
                p_df = p_df.append(p_df1[['ticker','datetime',price_name]])
            else:
                p_df = p_df.append(p_df1.drop(['Volume','Unadjclose'],axis=1))
    if p_df.shape==(0,0):
        warnings.warn("Data not found")
    else:
        p_df['datetime'] = pd.to_datetime(p_df['datetime'])
        p_df = p_df.sort_values(by=['ticker','datetime'])
    return p_df.reset_index(drop=True)

""" Feature construction """
def construct_features(p_df,price_name,**kwargs):
    feature_df = pd.DataFrame()

    feature_df['Return1'] = p_df[price_name].pct_change(1)
    feature_df['Return2'] = p_df[price_name].pct_change(2)
    feature_df['Return5'] = p_df[price_name].pct_change(5)
    feature_df['Return10'] = p_df[price_name].pct_change(10)
    
    feature_df['Ret1_lag1'] = feature_df['Return1'].shift(1)
    feature_df['Ret1_lag2'] = feature_df['Return1'].shift(2)
    feature_df['Ret1_lag3'] = feature_df['Return1'].shift(3)
    feature_df['Ret1_lag4'] = feature_df['Return1'].shift(4)
    feature_df['Ret1_lag5'] = feature_df['Return1'].shift(5)
    
    feature_df['Ret1_lag10'] = feature_df['Return1'].shift(10)
    feature_df['Ret1_lag20'] = feature_df['Return1'].shift(20)

    feature_df['Std5'] = p_df[price_name].pct_change().rolling(5).std()
    feature_df['Std10'] = p_df[price_name].pct_change().rolling(10).std()
    feature_df['Std20'] = p_df[price_name].pct_change().rolling(20).std()
    
    feature_df['MOM5'] = MOM(p_df[price_name], timeperiod=5)
    feature_df['MOM10'] = MOM(p_df[price_name], timeperiod=10)
    
    feature_df['EMA5'] = EMA(p_df[price_name], timeperiod=5)
    feature_df['EMA10'] = EMA(p_df[price_name], timeperiod=10)
    feature_df['EMA20'] = EMA(p_df[price_name], timeperiod=20)
    
    feature_df['MA5'] = MA(p_df[price_name], timeperiod=5)
    feature_df['MA10'] = MA(p_df[price_name], timeperiod=10)
    feature_df['MA20'] = MA(p_df[price_name], timeperiod=20)
    
    feature_df['weekday'] = p_df['datetime'].apply(lambda x:x.weekday())
    feature_df['month'] = p_df['datetime'].apply(lambda x:x.month)

    if 'benchmark' in kwargs.keys():
        # construct label of prediction for return against benchmark
        
        ret1 = p_df[price_name].shift(-1)/p_df[price_name] -1
        ret2 = kwargs['benchmark'].shift(-1)/kwargs['benchmark']-1
        feature_df['for_target'] = int(ret1>ret2)
    else:
        # construct label of prediction for 1 day return
        feature_df['for_target'] = p_df[price_name].shift(-1)/p_df[price_name] -1
   
    # set time and ticker
    feature_df = pd.concat([p_df[['ticker','datetime']],feature_df], axis=1)
    #feature_df['ticker'] = p_df['ticker']

    return feature_df

""" Data Preprocessing """       
def bulk_process(price_df,price_name='Open',start=None, end=None, **kwargs):
    """
    :Parameters:
        price_df : DataFrame          
        start : string
          The starting date of dataset
        end : string
          The ending date of dataset
    :Returns:
        X_train, Y_train, X_val, Y_val: numpy
   
    """
    """ Filter data"""
    p_df = price_df.copy()
    if start is not None:
        p_df = p_df[p_df['datetime']>=start]
    if end is not None:
        p_df = p_df[p_df['datetime']<=end]
       
    """ Construct features"""
    #TODO: if 'benchmark' in kwargs.keys():
    f_df = p_df.groupby('ticker').apply(lambda x:construct_features(x,price_name, **kwargs))
    weekday = pd.get_dummies(f_df['weekday'],drop_first=True)
    month = pd.get_dummies(f_df['month'],drop_first=True)
    f_df = pd.concat([f_df,weekday,month],axis = 1)
    f_df.drop(['weekday','month'],axis=1,inplace=True)
  
    f_df.dropna(inplace=True) #Attention: last day is dropped
    
    X = f_df.drop(['ticker','datetime','for_target'],axis=1).values
    Y = f_df['for_target'].values
    
    dt_map = f_df.iloc[:,[0,1]].reset_index(drop=True)
    
    return (X, Y, dt_map)

def get_datelist(dt_map):
    start = dt_map['datetime'].min()
    end = dt_map['datetime'].max()
    for ticker in dt_map['ticker'].unique():
        if dt_map.loc[dt_map['ticker']==ticker,'datetime'].min()==start and dt_map.loc[dt_map['ticker']==ticker,'datetime'].max()==end:
            date_list = dt_map.loc[dt_map['ticker']==ticker,'datetime'].tolist()
            break
    return date_list

def train_test_split(nTrain, nTest, dt_map, start=0, window = 'single', w = None):
    if window =='single':
        assert w is None,"cannot assign w for single window"
    elif window =='roll' or window =='expand':
        assert w is not None, "should assign w for rolling and expanding windows"
        assert w<=nTest,"w should be smaller than nVal"
    
    date_list = get_datelist(dt_map)   
    N = len(date_list)
    n = nTrain+nTest
    assert N>=n,"not enough dates"
    
    sel_date = date_list[start:start+n]
                    
    idx = []
    if window == 'single':
        d_train = sel_date[:nTrain]
        d_test = sel_date[nTrain:]
        idx_train = dt_map[dt_map['datetime'].isin(d_train)].index.values.tolist()
        idx_test = dt_map[dt_map['datetime'].isin(d_test)].index.values.tolist()
        idx.append((idx_train, idx_test))
    else: 
        for d in range(int(nTest/w)):
            if window =='roll':
                d_train = sel_date[(0 + d*w): (nTrain + d*w)]
            else:
                d_train = sel_date[0: (nTrain + d*w)]
            d_test = sel_date[nTrain+d*w:nTrain+(d+1)*w]
            idx_train = dt_map[dt_map['datetime'].isin(d_train)].index.values.tolist()
            idx_test = dt_map[dt_map['datetime'].isin(d_test)].index.values.tolist()
            idx.append((idx_train, idx_test))
        if (d+1)*w<nTest:
            d = d+1
            if window =='roll':
                d_train = sel_date[(0 + d*w): (nTrain + d*w)]
            else:
                d_train = sel_date[0: (nTrain + d*w)]
            d_test = sel_date[nTrain+d*w:n]
            idx_train = dt_map[dt_map['datetime'].isin(d_train)].index.values.tolist()
            idx_test = dt_map[dt_map['datetime'].isin(d_test)].index.values.tolist()
            idx.append((idx_train, idx_test))
    return idx    

def calc_datepoints(dt_map, start,split,end):
    date_list = get_datelist(dt_map)
    if not isinstance(start,datetime):
        start = datetime.strptime(start,'%Y-%m-%d')
    if not isinstance(end,datetime):
        end = datetime.strptime(end,'%Y-%m-%d')
    if not isinstance(split,datetime):
        split = datetime.strptime(split,'%Y-%m-%d')
    
    assert start < split 
    assert split < end   
    
    try:
        n_start = date_list.index(start)
    except ValueError:
        n_start = [ n for n,i in enumerate(date_list) if i>start ][0]    
    
    try:
        n_end = date_list.index(end)
    except ValueError:
        n_end = [ n for n,i in enumerate(date_list) if i<end ][-1]
        
    try:
        n_split = date_list.index(split)
    except ValueError:
        n_split = [ n for n,i in enumerate(date_list) if i>split ][0]  
    
    nTrain = n_split-n_start
    nTest = n_end-n_split+1
    
    return nTrain, nTest

def get_price_by_date(df, dt, n_sample):
    df = df.sort_values(by='datetime').reset_index(drop=True)
    try:
        n = df[df['datetime']==dt].index.values[0]
    except IndexError:
        n = df[df['datetime']<dt].index.values.max()
    sn = n-n_sample
    if sn<0:
        warnings.warn("Insufficient samples before the given date, will delete the ticker")
        sn=n+1
    a = np.empty(n_sample-n if n_sample>n else 0)
    a.fill(np.nan)#impute mising values with nan
    return df.iloc[sn:n+1,-1].values#include dt
    #np.append(a,df.iloc[sn:n+1,-1].values) 

def price_to_ret(price):
    return np.diff(np.log(price))
    
def calc_return_matrix(df_price, dt, n_sample, ticker_list):
    price = df_price[df_price['ticker'].isin(ticker_list)]
    price = price.groupby('ticker').apply(get_price_by_date,dt,n_sample)
    price = price[ticker_list]
    deleted = price[price.apply(len) <= 0].index.values.tolist()
    price = price[price.apply(len) > 0]
    ret = price.apply(price_to_ret)
    ret = np.stack(ret)
    return ret,deleted
