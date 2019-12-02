# Enhanced Index Tracking Strategy with application of Machine Learning and Tracking Error Optimization.

## Introduction
This project constructs an Enhanced Index Tracking portfolio which aims to track the benchmark index closely and generate excess return by active management. 
We construct machine learning algorithms, e.g. Logistic Regression, SVM, Random Forest and XGBoost, to select stocks from historical S&P500 constituents and optimize value weights through tracking error optimization. 
According to the backtesting result in the US equity market from 2014-1-2 to 2019-8-3, the constructed portfolio achieves 64.1% accumulative return and 0.04 alpha.

## Environment Set Up
### Python Version:
3.6.7

### Required Modules:
* numpy
* pandas
* scipy
* sklearn
* xgboost
* talib
* cvxopt
* matplotlib
* time
* os
* warnings
* datetime

## Instructions
The [\_\_datacache\_\_](__datacache__) folder includes adjusted daily price for SPY and sample stock tickers.

The [main.ipynb](main.ipynb) notebook showcases the data preprocessing, EDA, stock selection and portfolio construction processes. 
The [main.py](main.py) file contains equivalent scripts.

The [main_parameter_tuning.py](main_parameter_tuning.py) file runs grid search for paramter tuning with Monte Carlo experiments.

[data_pipeline.py](data_pipeline.py), [train.py](train.py), [portfolio_generator.py](portfolio_generator.py) are customized sub-modules for the project.
