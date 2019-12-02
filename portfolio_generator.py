#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:33:33 2019

@author: Sherry
"""

"""Import Modules"""
## Numerical Computation
import numpy as np  
from data_pipeline import train_test_split, calc_return_matrix
from sklearn import preprocessing
## Model training
from train import ModelClass
## Optimization
import cvxopt as opt  
from cvxopt import blas, solvers
from sklearn.covariance import LedoitWolf

import time 

solvers.options['show_progress'] = False


def stock_selection(X, Y, dt_map, nTrain, nTest, model_type, method='single', w=None, verbose = 0, **kwargs):
    assert X.shape[0]==len(Y), "X and Y should have the same shape"
    idx_list = train_test_split(nTrain, nTest, dt_map, window = method, w = w)
    idx_test_roll = [y for x in idx_list for y in x[1]]
    Y_true = Y[idx_test_roll]
    Y_pred = np.array([])
    start = time.time()
    for d, (idx_train,idx_test) in enumerate(idx_list):
        if not verbose:
            print("---- Now processing Window %d/%d -----"%(d+1, len(idx_list)))
        X_train, X_test = X[idx_train],X[idx_test]
        Y_train = Y[idx_train]
        if model_type !='rf':
            scaler = preprocessing.StandardScaler()
            X_train = scaler.fit_transform(X_train)     
            X_test = scaler.transform(X_test)     
        model = ModelClass(model_type,**kwargs)
        model.fit(X_train, Y_train)
        Y_pred = np.append(Y_pred,model.predict(X_test))
    print("---- Finished in %.2f seconds"%(time.time()-start))
    score = model.cal_score(Y_true,Y_pred)
    df_stock = dt_map.iloc[idx_test_roll,:]
    df_stock['select'] = Y_pred.tolist()
    df_stock = df_stock[df_stock['select']==1]
    df_stock.drop('select',axis=1,inplace=True)
    return df_stock.sort_values(by=['datetime','ticker']), score 
    
def weight_opt(returns,benchmark, lower = 0, upper = 1, ph=2**7, cov_method='sample', seed = 123):
    np.random.seed(seed)
    n_asset, n_sample = returns.shape
    rets = np.asmatrix(returns)  
    #N = 10
    #phs = [2**(t-2) for t in range(N)]  
    # Convert to cvxopt matrices 
    if cov_method == 'sample':
        Cov = opt.matrix(np.cov(rets,benchmark))
    elif cov_method == 'lw':
        Cov = opt.matrix(LedoitWolf().fit(np.append(np.transpose(rets),benchmark.reshape(n_sample,1), axis=1)).covariance_)
    else:
        raise ValueError('cov_method should be in {}'.format({'sample', 'lw'}))
    S = Cov[:n_asset,:n_asset]
    r_mean = opt.matrix(np.nanmean(rets, axis=1)) # n*1
    Cb = Cov[:n_asset,n_asset]
    # Create constraint matrices  
    G = opt.matrix(np.append(np.eye(n_asset),-np.eye(n_asset),axis = 0))   # 2n x n identity matrix  
    h = opt.matrix(np.append(upper*np.ones((n_asset,1)),-lower*np.ones((n_asset,1)),axis = 0)) 
    A = opt.matrix(1.0, (1, n_asset))  
    b = opt.matrix(1.0)  
    # Calculate efficient frontier weights using quadratic programming  
    x = solvers.qp(ph*S, -ph*Cb-r_mean, G, h, A, b)['x']
    #portfolios = [solvers.qp(ph*S, -ph*Cb-r_mean, G, h, A, b)['x']  
    #              for ph in phs]  
    # CALCULATE RISKS AND RETURNS FOR FRONTIER  
    ret = blas.dot(r_mean, x)
    #[blas.dot(r_mean, x) for x in portfolios]  
    errors = blas.dot(x, S*x)+Cov[n_asset,n_asset]-2*blas.dot(Cb,x)
    #[blas.dot(x, S*x)+Cov[n_asset,n_asset]-2*blas.dot(Cb,x) for x in portfolios]  
    return np.transpose(np.array(x))[0], ret, errors#, ret_opt, risk_opt   


def satellite_portfolio(df_stock, price, benchmark, n_sample, strategy='avg', verbose = 0, **kwargs):
    #df_stock.groupby('datetime')['ticker']
    df_port = df_stock.copy()
    df_port['weight']=0
    for dt in df_port['datetime'].unique():
        dt = np.datetime64(dt,'D')
        ticker_list = df_port.loc[df_port['datetime']==dt,'ticker'].tolist()        
        if not verbose:
            #print(dt,ticker_list)
            print(dt)
        if strategy == 'avg':
            weights = 1/len(ticker_list)
            deleted = []
        elif strategy == 'EIT':
            if len(ticker_list)==1:
                weights = 1
                deleted = []
            else:
                ret,deleted = calc_return_matrix(price, dt, n_sample, ticker_list)
                if ret.shape[0]==1:
                    weights = 1
                else:
                    bk = calc_return_matrix(benchmark,dt,n_sample,['SPY'])[0]
                    weights = weight_opt(ret, bk, **kwargs)[0]
        else:
            raise ValueError('strategy should be in {}'.format({'avg', 'EIT'}))
        df_port.loc[(df_port['datetime']==dt)&(df_port['ticker'].isin(deleted)),'weight']=0            
        df_port.loc[(df_port['datetime']==dt)&(~df_port['ticker'].isin(deleted)),'weight']=weights            
    return df_port