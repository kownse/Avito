#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 20:25:54 2018

@author: kownse
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold, StratifiedKFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from datetime import datetime
import kaggle_util

NFOLDS = 5
SEED = 2018
VALID = False
debug = 0

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
    
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

if __name__ == "__main__":
    
    frm = 0
    to = 1503424
    if debug:    
        frm = 0
        to = 10000
    
    print("\nData Load Stage")
    training = pd.read_csv('../input/train.csv', skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
    traindex = training.index
    testing = pd.read_csv('../input/test.csv', skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
    testdex = testing.index
    
    ntrain = training.shape[0]
    ntest = testing.shape[0]
    
    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    
    y = training.deal_probability.copy()
    training.drop("deal_probability",axis=1, inplace=True)
    print('Train shape: {} Rows, {} Columns'.format(*training.shape))
    print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
    
    print("Combine Train and Test")
    df = pd.concat([training,testing],axis=0)
    del training, testing
    gc.collect()
    print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
    
    
    print("Feature Engineering")
    df["price"] = np.log(df["price"]+0.001)
    df["price"].fillna(df.price.mean(),inplace=True)
    df["image_top_1"].fillna(-999,inplace=True)
    
    print("\nCreate Time Variables")
    df["Weekday"] = df['activation_date'].dt.weekday
    #df["Weekd of Year"] = df['activation_date'].dt.week
    #df["Day of Month"] = df['activation_date'].dt.day
    
    # Create Validation Index and Remove Dead Variables
    training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
    validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
    df.drop(["activation_date","image"],axis=1,inplace=True)
    
    print("\nEncode Variables")
    categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
    print("Encoding :",categorical)
    
    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        df[col].fillna('Unknown')
        df[col] = lbl.fit_transform(df[col].astype(str))
        
    print("\nText Features")
    
    # Feature Engineering 
    
    # Meta Text Features
    textfeats = ["description", "title"]
    df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    
    df['title'] = df['title'].apply(lambda x: cleanName(x))
    df["description"]   = df["description"].apply(lambda x: cleanName(x))
    
    for cols in textfeats:
        df[cols] = df[cols].astype(str) 
        df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
        df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment)) # Count number of Letters
    
    df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']
    
    print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
    russian_stop = set(stopwords.words('russian'))
    
    tfidf_para = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        #"min_df":5,
        #"max_df":.9,
        "smooth_idf":False
    }
    
    
    def get_col(col_name): return lambda x: x[col_name]
    ##I added to the max_features of the description. It did not change my score much but it may be worth investigating
    vectorizer = FeatureUnion([
            ('description',TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=17000,
                **tfidf_para,
                preprocessor=get_col('description'))),
            ('title',CountVectorizer(
                ngram_range=(1, 2),
                stop_words = russian_stop,
                #max_features=7000,
                preprocessor=get_col('title')))
        ])
        
    
    #Fit my vectorizer on the entire dataset instead of the training rows
    #Score improved by .0001
    vectorizer.fit(df.to_dict('records'))
    
    ready_df = vectorizer.transform(df.to_dict('records'))
    tfvocab = vectorizer.get_feature_names()
    
    # Drop Text Cols
    textfeats = ["description", "title"]
    df.drop(textfeats, axis=1,inplace=True)
    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                    'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
    
    #Ridge oof method from Faron's kernel
    #I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
    #It doesn't really add much to the score, but it does help lightgbm converge faster
    ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
    ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])
    
    rms = sqrt(mean_squared_error(y, ridge_oof_train))
    print('Ridge OOF RMSE: {}'.format(rms))
    
    print("Modeling Stage")
    
    ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
    
    df['ridge_preds'] = ridge_preds
    
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 270,# 37,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.75,
        'learning_rate': 0.016,
        'nthread': 6,
        'verbose': 0,
        } 
    
    tfvocab = df.columns.tolist() + tfvocab
    testing = hstack([csr_matrix(df[ntrain:].values),ready_df[ntrain:]])
    
    nfold = 5
    skf = StratifiedKFold(y, n_folds=nfold)
    
    for i, (train_split, val_split) in enumerate(skf):
        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=5)

        X_train = hstack([csr_matrix(df.iloc[train_split].values),ready_df[train_split]])
        X_valid = hstack([csr_matrix(df.iloc[val_split].values),ready_df[val_split]]) # Sparse Matrix 
        y_train = y[train_split] 
        y_valid = y[val_split]
        
        lgtrain = lgb.Dataset(X_train, y_train,
                        feature_name=tfvocab,
                        categorical_feature = categorical)
        lgvalid = lgb.Dataset(X_valid, y_valid,
                        feature_name=tfvocab,
                        categorical_feature = categorical)
        
        lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=26000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        print("Model Evaluation Stage")
        rmse = np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid)))
        print('RMSE:', rmse)
        
        str_now = datetime.now().strftime("%m-%d-%H-%M")
        if not debug:
            lgb_clf.save_model('../model/ridge_{}.txt'.format(i), lgb_clf.best_iteration)
        else:
            lgb_clf.save_model('../model/ridge_debug_{}.txt'.format(i), lgb_clf.best_iteration)
       
        lgpred = lgb_clf.predict(testing, num_iteration = lgb_clf.best_iteration)
        lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
        lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
        
        subfile = '../result/ridge_{}.csv'.format(i)
        if debug:
            subfile = '../result/ridge_debug{}.csv'.format(i)
        kaggle_util.save_result(lgsub, subfile, competition = 'avito-demand-prediction', send = False, index = True)
    
    result_list = []
    for i in range(nfold):
        subfile = '../result/ridge_{}.csv'.format(i)
        if debug:
            subfile = '../result/ridge_debug{}.csv'.format(i)
        result_list.append((subfile, 1 / nfold))
        
    kaggle_util.ensemble(result_list, not debug, 
                         competition = 'avito-demand-prediction', 
                         score_col = 'deal_probability')
    