#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 01:30:43 2019

@author: Sherry
"""

""" check portfolio data, comparing with Quantopian"""

import pandas as pd

file_name_list = ['port_avg_roll_l1_thr0.005_fill',#wrong, updated
                  'port_avg_roll_l1_thr0.0025_fill',#wrong, updated
                  'port_eit_lw_roll_l1_thr0.005_fill',
                  'port_eit_lw_roll_l1_thr0.0025_fill',
                  'port_eit_roll_l1_thr0.005_fill',
                  'port_eit_roll_l1_thr0.0025_fill'#wrong,updated
                  ]

#for file in file_name_list:
file = file_name_list[3]
df = pd.read_csv('__datacache__/'+file+'.csv')
df_ = pd.read_csv('__datacache__/'+file_name_list[5]+'.csv')

df.shape
df.groupby('datetime')['ticker'].count().min()
df.groupby('datetime')['ticker'].count().max()

df1 = df[df['weight']<200]
df1 = df1.sort_values(['datetime','ticker'])
df1.shape
#df1.groupby('datetime')['ticker'].count()
df1.groupby('datetime')['weight'].sum()
df1.groupby('datetime')['weight'].sum().min()
df1.groupby('datetime')['weight'].sum().max()