#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:25:57 2019

@author: Sherry
"""

"""Import Modules"""
# Numerical Computation
import numpy as np
import pandas as pd
# Data Processing
from data_pipeline import *
from scipy.stats.stats import pearsonr, kendalltau
from sklearn import preprocessing
## Training
from train import ModelClass
## Porfolio Construction
from portfolio_generator import stock_selection, satellite_portfolio
## Visualization
import matplotlib.pyplot as plt

"""**************Hyperparameters************"""
path = r'../data/adj_daily'
start = '1997-12-31'
end = '2019-8-5' #'2019-11-5'
split = '2014-1-1'
nTrain = 8*12*20 # 5 years 
nVal = 1*12*20 # 1 year
thr = 0.005

#start = datetime.strptime(start,'%Y-%m-%d')
#end = datetime.strptime(end,'%Y-%m-%d')

"""**************Loading Data*****************"""
print("- Loading Data")

# uncomment if need to load raw price data
#price_df = load_data(path,'Open')
#price_df.to_csv('__datacache__/price_open_df.csv',index=False)

# stock prices
price_df = pd.read_csv('__datacache__/sample_stock_data.csv')
price_df['datetime'] = pd.to_datetime(price_df['datetime'])

# benchmark prices
benchmark_df = pd.read_csv('__datacache__/SPY_adj.csv')[['date','Open']]
benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
benchmark_df = benchmark_df.rename(columns = {'date':'datetime'})
benchmark_df.insert(0, 'ticker', 'SPY')


"""**************Data Preprocessing**************"""
print('-Data Preprocessing')
X, Y, dt_map= bulk_process(price_df, 'Open', start, end)
Y_label = (Y>thr).astype(int)
Y_label.sum()/len(Y)

NTrain, NTest = calc_datepoints(dt_map, start, split, end)

# train-test split
idx = train_test_split(NTrain, NTest, dt_map, start=0, window = 'single')
for d, (idx_train, idx_test) in enumerate(idx):
    X_train, X_test = X[idx_train],X[idx_test]
    Y_train_label, Y_test_label = Y_label[idx_train],Y_label[idx_test]
    train_map = dt_map.iloc[idx_train,:].reset_index(drop=True)
    test_map = dt_map.iloc[idx_test,:].reset_index(drop=True)

## Standarization if necessary
#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)     
#X_test = scaler.transform(X_test)  

# benchmark data
date_list = get_datelist(dt_map)
benchmark_df = benchmark_df.iloc[
        benchmark_df[benchmark_df['datetime']==min(date_list)].index.values[0]-1:
            benchmark_df[benchmark_df['datetime']==max(date_list)].index.values[0]+1,:]
benchmark_df = benchmark_df.reset_index(drop=True)
    
# check date matching
pd.DataFrame({'stock_flag':np.ones(len(date_list))},
              index=date_list).join(benchmark_df.set_index('datetime'),how='outer').isna().sum()
    
"""**************Data Exploration***************"""
plt.hist(Y, color = 'blue', edgecolor = 'black',
         bins = 800)
plt.title('Histogram of Labels')
plt.xlim((-.25,.25))

pcor = [pearsonr(X[:,i],Y)[0] for i in range(X.shape[1])]
tcor = [kendalltau(X[:,i],Y).correlation for i in range(X.shape[1])]

"""**************Training Model**************"""
X_tr = X_train[:1000]
X_ts = X_train[1000:100+1000]
Y_tr = Y_train_label[:1000]
Y_ts = Y_train_label[1000:100+1000]

# logistic regression
lr = ModelClass(model_type='lr',solver='saga',penalty='l1',C=10)
lr.fit(X_tr,Y_tr)
y_pred = lr.predict(X_ts)
print('*****train scores*****')
scores_test = lr.evaluate(X_tr, Y_tr)
print('*****test scores*****')
scores_train = lr.evaluate(X_ts, Y_ts)

# svc
svc = ModelClass(model_type='svc',kernel='rbf',C=3,gamma=.09)
svc.fit(X_tr,Y_tr)
y_pred = svc.predict(X_ts)
print('*****train scores*****')
scores_test = svc.evaluate(X_tr, Y_tr)
print('*****test scores*****')
scores_train = svc.evaluate(X_ts, Y_ts)


#random forest
rf = ModelClass(model_type='rf',
                n_estimators=50, 
                oob_score=True, 
                max_depth=10, 
                min_samples_split=5, 
                min_samples_leaf=5,
                random_state=1234)
rf.fit(X_tr,Y_tr)
y_pred = rf.predict(X_ts)
print('*****train scores*****')
scores_test = rf.evaluate(X_tr, Y_tr)
print('*****test scores*****')
scores_train = rf.evaluate(X_ts, Y_ts)


# xgboost
xgb = ModelClass(model_type='xgb')
xgb.fit(X_tr,Y_tr)
y_pred = xgb.predict(X_ts)
print('*****train scores*****')
scores_test = xgb.evaluate(X_tr, Y_tr)
print('*****test scores*****')
scores_train = xgb.evaluate(X_ts, Y_ts)

"""***************Parameter Tuning****************"""
# see main_parameter_tuning.py

"""******************** Stock Selection ******************"""
# parameters
model_type = 'rf'
w=60

# rolling window predict
df_stock3, score3 = stock_selection(X, Y_label, dt_map, int(NTrain), NTest,
                                  model_type, 
                                  n_estimators = 140, 
                                  max_depth=5, 
                                  min_samples_split=5, 
                                  min_samples_leaf=5,
                                  n_jobs = -1,
                                  random_state = 1234,
                                  class_weight = {0:1,1:1.25},
                                  method='roll', 
                                  w=w, 
                                  verbose = 0)

score = score3
print("confusion matrix:\n",score[0])
print("Accuracy:%f\nPrecision:%f\nRecall:%f\nF1:%f\nTPR:%f\nFPR:%f"%\
      (score[1],score[2],score[3],score[4],score[5],score[6]))


# single window predict
df_stock4, score4 = stock_selection(X, Y_label, dt_map, NTrain, NTest,
                                  model_type, 
                                  n_estimators = 140, 
                                  max_depth=5, 
                                  min_samples_split=5, 
                                  min_samples_leaf=5,
                                  n_jobs = -1,
                                  random_state = 1234,
                                  class_weight = {0:1,1:1.25},
                                  method='single', verbose = 0)

# save the result of stock selection
df_stock3.to_csv('__datacache__/df_stock_roll_lab2.csv',index=False)
df_stock4.to_csv('__datacache__/df_stock_sing_lab2.csv',index=False)

"""******************* Construct Portfolio ****************"""

## load data of stock selection
# df_stock = pd.read_csv("__datacache__/df_stock_roll.csv")
# df_stock['datetime']=pd.to_datetime(df_stock['datetime'])
# # evaluate if necessary
# pred = df_stock.copy()
# pred['select']=1
# pred = pd.merge(test_map,pred,on=['ticker','datetime'],how='left')
# pred = pred.fillna(0)
# score = rf.cal_score(Y_test_label,pred['select'])

# satellite portfolio
n_sample = 200
ph = 2**7
lower = 0.001
upper = 0.6

port1 = satellite_portfolio(df_stock, price_df, benchmark_df, n_sample, strategy = 'avg',verbose=1)
port2 = satellite_portfolio(df_stock, price_df, benchmark_df, n_sample, strategy = 'EIT',
                            ph = ph, lower = lower, upper = upper,verbose=0)

# overall portfolio
port = port1
df = pd.DataFrame({'ticker':['SPY']*len(date_list),
                   'datetime':date_list,
                   'weight':[0.8]*len(date_list)})#80% of SPY
df = df[df['datetime']>split]
port['weight'] = port['weight']*0.2 # 20% of satellite portfolio
port = port.append(df)
port['datetime'] = port['datetime'].apply(lambda x: np.datetime64(x,'D'))
port = port.sort_values(by=['datetime','ticker'])

# fill ticker to meet the requirement of backtestig platform
unique_ticker = port['ticker'].unique()
port_fill = port.set_index(['ticker','datetime']).unstack(level = 'datetime').reindex(unique_ticker).stack(level='datetime',dropna=False).reset_index()
port_fill = port_fill.fillna(99999)#fill the weights of unselected tickers with an extremely large value

# save contructed portfolios
port_fill.to_csv("__datacache__/port_avg_roll_l1_thr0.0025_fill.csv",index = False)

# How to use port_fill
port_fill[port_fill['weight']<200]
# check
port_fill[port_fill['weight']<200].groupby('datetime')['weight'].sum()