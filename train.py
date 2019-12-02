#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:30:39 2019

@author: Sherry
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:28:13 2019

@author: Sherry

Run Monte Carlo Experiments to train machine learning models with rolling windows
"""

"""Import Modules"""
# Numerical Computation
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score,average_precision_score
from data_pipeline import get_datelist, train_test_split
# Plot
import matplotlib.pyplot as plt

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#from keras.utils import to_categorical
#from keras import regularizers
#from keras.models import Model

import time

class ModelClass:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        if model_type == 'lr':
            model = LogisticRegression(**kwargs)         
            self.model = model            
        elif model_type == 'svc':
            model = SVC(**kwargs)
            self.model = model
        elif model_type == 'rf':
            model = RandomForestClassifier(**kwargs)
            self.model = model
        elif model_type == 'xgb':
            model = XGBClassifier(**kwargs)
            self.model = model
        ##elif model_type =='mlp':
            ## TODO           
        else:
            raise ValueError('model_type should be in {}'.format({'lr','svc','rf','xgb'}))
        #for key,value in kwargs:
                  
    def fit(self, X_train, Y_train,**kwargs):
        #self.X_train = X_train
        #self.Y_train = Y_train
        self.model.fit(X_train,Y_train,**kwargs)
    
    def predict(self, X_test): 
        #self.X_test = X_test
        #self.Y_pred = self.model.predict(X_test)
        return self.model.predict(X_test)#self.Y_pred
    
    def evaluate(self, X_test, Y_test,verbose=0):
        Y_pred = self.model.predict(X_test)
        if self.model_type =='svc':
            Y_pred_prob = self.model.decision_function(X_test)
        elif self.model_type in ['lr','rf','xgb']:
            Y_pred_prob = self.model.predict_proba(X_test)[:,1]
        TN,FP,FN,TP = confusion_matrix(Y_test,Y_pred).ravel()
        # Accuracy
        Acc = (TP+TN)/(TP+TN+FP+FN)
        # Precision
        Prec = TP/(TP+FP) 
        Avg_prec = average_precision_score(Y_test,Y_pred_prob)
        #Recall
        Rec = TP/(TP+FN)
        # F1-score
        F1 = 2*Prec*Rec/(Prec+Rec)
        # TPR
        TPR = TP/(TP+FN)
        # FPR
        FPR = FP/(FP+TN)
        # AUC
        AUC = roc_auc_score(Y_test, Y_pred_prob)
        
        if not verbose:
            print("Accuracy:%f\nPrecision:%f\nAverage Precision:%f\nRecall:%f\nF1:%f\nTPR:%f\nFPR:%f\nAUC:%f"%\
                  (Acc, Prec, Avg_prec, Rec, F1, TPR, FPR, AUC))

        return Acc, Prec, Avg_prec, Rec, F1, TPR, FPR, AUC
    
    def cal_score(self,Y_true, Y_pred):
        cf_matrix = confusion_matrix(Y_true,Y_pred)
        TN,FP,FN,TP = cf_matrix.ravel()
        # Accuracy
        Acc = (TP+TN)/(TP+TN+FP+FN)
        # Precision
        Prec = TP/(TP+FP) 
        #Recall
        Rec = TP/(TP+FN)
        # F1-score
        F1 = 2*Prec*Rec/(Prec+Rec)
        # TPR
        TPR = TP/(TP+FN)
        # FPR
        FPR = FP/(FP+TN)

        return cf_matrix, Acc, Prec, Rec, F1, TPR, FPR
        

            
class MonteCarlo:
    """
    Implment Monte Carlo simulation, return average precision and accuracy
    Input: Selected Model, datasets and n_folds
    Output: average precsion and accuracy(train and test)
    
    """    
    def __init__(self, model, X, Y, nTrain, nVal, nReps=10, window = 'roll', w=None, seed=1234, verbose = 0):
        
        """ Check and assign inputs
        :Parameters:
          model : Model
            selected model to train
          X : numpy.ndarray
            attribute data
          Y : numpy.ndarray
            target data
          nTrain : int
            number of cases in training set
          nVal : int
            number of cases in validating set
          nReps : int
            number of repetitions of Monte Carlo experiments
          window : str {'single','roll','expand'}
            training method in a single repitition
          w : int
            length of steps to move forward in rolling window and expanding window
          seed : int
            random number generator seed
          verbose : int {0,1}  
            print or not, 1 = print, 0 = not print
        
        """
        
        assert X.shape[0]==Y.shape[0],"X and Y should have the same length"
        assert nTrain+nVal<X.shape[0],"data deficient"
        
        self.model = model
        self.X = X
        self.Y = Y
        self.nTrain = nTrain
        self.nVal = nVal
        self.nReps = nReps
        self.seed = seed
        self.window = window
        if window =='single':
            assert w is None,"cannot assign w for single window"
        elif window =='roll' or window =='expand':
            assert w is not None, "should assign w for rolling and expanding windows"
            assert w<=nVal,"w should be smaller than nVal"
            self.w = w
        else:
            raise ValueError('window should be in {}'.format({'single','roll','expand'}))
        self.verbose = verbose
        self.metrics = []
    
    def split(self, date_ticker_map):
        
        date_list = get_datelist(date_ticker_map)
        
        N = len(date_list)
        n = self.nTrain+self.nVal
        
        assert N>n,"not enough dates"
        
        np.random.seed(self.seed)
        ss = np.random.randint(0,N-n+1,self.nReps)
        idx = []
        for s in ss:
            sel_date = date_list[s:s+n]
            idx_1 = []
            if self.window == 'single':
                d_train = sel_date[:self.nTrain]
                d_val = sel_date[self.nTrain:]
                idx_train = date_ticker_map[date_ticker_map['datetime'].isin(d_train)].index.values.tolist()
                idx_val = date_ticker_map[date_ticker_map['datetime'].isin(d_val)].index.values.tolist()
                idx_1.append((idx_train, idx_val))
            else: 
                for d in range(int(self.nVal/self.w)):
                    if self.window =='roll':
                        d_train = sel_date[(0 + d*self.w): (self.nTrain + d*self.w)]
                    else:
                        d_train = sel_date[0: (self.nTrain + d*self.w)]
                    d_val = sel_date[self.nTrain+d*self.w:self.nTrain+(d+1)*self.w]
                    idx_train = date_ticker_map[date_ticker_map['datetime'].isin(d_train)].index.values.tolist()
                    idx_val = date_ticker_map[date_ticker_map['datetime'].isin(d_val)].index.values.tolist()
                    idx_1.append((idx_train, idx_val))
                if (d+1)*self.w<self.nVal:
                    d = d+1
                    if self.window =='roll':
                        d_train = sel_date[(0 + d*self.w): (self.nTrain + d*self.w)]
                    else:
                        d_train = sel_date[0: (self.nTrain + d*self.w)]
                    d_val = sel_date[self.nTrain+d*self.w:n]
                    idx_train = date_ticker_map[date_ticker_map['datetime'].isin(d_train)].index.values.tolist()
                    idx_val = date_ticker_map[date_ticker_map['datetime'].isin(d_val)].index.values.tolist()
                    idx_1.append((idx_train, idx_val))
            idx.append(idx_1)                               
        return idx
    
    def experiment(self, date_ticker_map):
        for i,idx_list in enumerate(self.split(date_ticker_map)): 
            if not self.verbose:
                print("----Now processing Repeat: %d----" % (i+1))
            start = time.time()
            ypred = np.array([])
            ytrue = np.array([])
            train_acc =[]
            train_prec = []
            for d, (idx_train,idx_val) in enumerate(idx_list):                
                if not self.verbose:
                    print("---- Now processing Window %d/%d -----"%(d+1, len(idx_list)))
                X_train, X_val = self.X[idx_train],self.X[idx_val]
                Y_train, Y_val = self.Y[idx_train],self.Y[idx_val]
                self.model.fit(X_train,Y_train)
                train_pred = self.model.predict(X_train)
                acc,prec = self.model.cal_score(Y_train, train_pred)[1:3]
                ypred = np.append(ypred,self.model.predict(X_val))
                ytrue = np.append(ytrue,Y_val)
                train_acc.append(acc)
                train_prec.append(prec)
            train_score = [np.mean(train_acc),np.mean(train_prec)]
            val_score = self.model.cal_score(ytrue,ypred)            
            self.metrics.append(val_score[1:])
            if not self.verbose:
                print("----Rep %d finished in %.2f seconds----\n train accuracy: %.4f, train precision: %.4f, val accuracy: %.4f, val precision: %.4f, true signal: %.4f" 
                      %( i+1, (time.time()-start),train_score[0], train_score[1], val_score[1], val_score[2], ytrue.sum()/len(ytrue)))
        return np.mean(self.metrics, axis = 0)   
