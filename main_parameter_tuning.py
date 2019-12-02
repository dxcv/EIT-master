#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:16:18 2019

@author: Sherry
"""

"""Import Modules"""
# Numerical Computation
import numpy as np
import pandas as pd
# Data Processing
from data_pipeline import *
from sklearn import preprocessing
## Training
from train import ModelClass, MonteCarlo
from sklearn.model_selection import ParameterGrid
import time 

"""**************Hyperparameters************"""
path = r'../../data/adj_daily'
start = '1997-12-31'
end = '2019-8-5' #'2019-11-5'
split = '2014-1-1'
nTrain = 8*12*20 # 8 years 
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
print("- Data Preprocessing")

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


"""***************Parameter Tuning****************"""
print("- Parameter Tuning")

# random forest
n_estimators = [40,60,80]#range(40,200,20)

max_depth = range(5,16,2)
min_samples_split = range(20,201,20)

#min_samples_split = range(80,150,20)
min_samples_leaf = range(10,60,10)

max_features = range(3,11,2)

param_grid={'n_estimators':n_estimators}
param_grid = list(ParameterGrid(param_grid))

window = 'roll'
w = 40
nReps = 5

scores_rf = pd.DataFrame()
best_scores_prec= 0
best_scores_acc = 0
start_time = time.time()
for j, param_set in enumerate(param_grid):
    print('----------Start Parameter Set: n_estimators=',param_set.get('n_estimators'),
          'max_depth=',param_set.get('max_depth'),
          'min_samples_split',param_set.get('min_samples_split'),'----------------')
    rf = ModelClass(model_type='rf',
                       n_estimators=param_set.get('n_estimators'), 
                       oob_score=True, 
                       max_depth=param_set.get('max_depth'), 
                       min_samples_split=param_set.get('min_samples_split'), 
                       min_samples_leaf=5,
                       max_features = "auto",
                       random_state=1234)
    rf_mc = MonteCarlo(rf, X_train, Y_train_label, nTrain, nVal, nReps = nReps, window=window, w=w, verbose = 0, seed =j)
    cur_score = rf_mc.experiment(train_map)
    tmp = pd.DataFrame(cur_score.reshape(1,6), columns = ['Accuracy','Precision','Recall', 'F1', 'TPR', 'FPR'])
    tmp['n_estimators']=param_set.get('n_estimators')
    scores_rf = scores_rf.append(tmp)
    if best_scores_acc < cur_score[0]:
        best_scores_acc = cur_score[0]
        best_param_acc = param_set
        
    if best_scores_prec < cur_score[1]:
        best_scores_prec = cur_score[1]
        best_param_prec = param_set
    print("******** average accuracy:",cur_score[0],"average precision:",cur_score[1])
print("best parameter set is %s, with Accuracy is %.4f" % (best_param_acc, best_scores_acc))
print("best parameter set is %s, with Precision is %.4f" % (best_param_prec, best_scores_prec))
print("finsied in %s seconds" %(time.time()-start_time))
