#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple avec XGBoost - Package XGBoost a importer

"""

import pandas as pd
import time
import numpy as np
import xgboost as xgb


from sklearn.cross_validation import train_test_split

start_time = time.time()

print('reading train file ...')

train = pd.read_pickle('train_dataframe.pkl')

y = pd.read_pickle('y_dataframe.pkl')

#print(train.head())
#print(y.head())

print('[{}] Start XGBoost Training'.format(time.time() - start_time))

#params = {'eta': 0.3,
#          'max_depth': 6,
#          'subsample': 0.8,
#          'colsample_bytree': 1,
#          'colsample_bylevel':0.7,
#          'min_child_weight':1,
#          'alpha':4,
#          'objective': 'binary:logistic',
#          'eval_metric': 'auc',
#          'random_state': 10,
#          'silent': True}

params= {'nthread': -1,
         'max_depth':10,
         'learning_rate':0.025,
         'eval_metric': 'auc',
         'silent':True,
         'subsample':0.8,
         'colsample_bytree':0.8}

x1, x2, y1, y2 = train_test_split(train, y, test_size=0.2, random_state=99)

del train, y

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

model = xgb.train(params, xgb.DMatrix(x1, y1),
                  100,
                  watchlist,
                  verbose_eval=1)

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

del x1, x2, y1, y2

print('reading test file ...')

test = pd.read_pickle('test_dataframe.pkl')

print(test.head())

#%%
sub = pd.read_pickle('sub_dataframe.pkl')

print(sub.head())

sub['is_attributed'] = model.predict(xgb.DMatrix(test),
   ntree_limit=model.best_ntree_limit)

sub.to_csv('xgb_sub.csv',index=False)
