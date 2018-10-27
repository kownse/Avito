#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:55:12 2018

@author: kownse
"""

# Light GBM for Avito Demand Prediction Challenge
# Uses Bag-of-Words, meta-text features, and dense features.
# NO COMPUTER VISION COMPONENT.

# https://www.kaggle.com/c/avito-demand-prediction
# By Nick Brooks, April 2018

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
#print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import kaggle_util
from calcImgAtt import load
import parse_att
from tqdm import tqdm
from datetime import datetime
import string

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation as LDA

dict_att_path = '../input/dict_imgatt.pkl'
NFOLDS = 5
SEED = 5

debug = 0

parent_category_name_map = {"Личные вещи" : "Personal belongings",
                            "Для дома и дачи" : "For the home and garden",
                            "Бытовая электроника" : "Consumer electronics",
                            "Недвижимость" : "Real estate",
                            "Хобби и отдых" : "Hobbies & leisure",
                            "Транспорт" : "Transport",
                            "Услуги" : "Services",
                            "Животные" : "Animals",
                            "Для бизнеса" : "For business"}

region_map = {"Свердловская область" : "Sverdlovsk oblast",
            "Самарская область" : "Samara oblast",
            "Ростовская область" : "Rostov oblast",
            "Татарстан" : "Tatarstan",
            "Волгоградская область" : "Volgograd oblast",
            "Нижегородская область" : "Nizhny Novgorod oblast",
            "Пермский край" : "Perm Krai",
            "Оренбургская область" : "Orenburg oblast",
            "Ханты-Мансийский АО" : "Khanty-Mansi Autonomous Okrug",
            "Тюменская область" : "Tyumen oblast",
            "Башкортостан" : "Bashkortostan",
            "Краснодарский край" : "Krasnodar Krai",
            "Новосибирская область" : "Novosibirsk oblast",
            "Омская область" : "Omsk oblast",
            "Белгородская область" : "Belgorod oblast",
            "Челябинская область" : "Chelyabinsk oblast",
            "Воронежская область" : "Voronezh oblast",
            "Кемеровская область" : "Kemerovo oblast",
            "Саратовская область" : "Saratov oblast",
            "Владимирская область" : "Vladimir oblast",
            "Калининградская область" : "Kaliningrad oblast",
            "Красноярский край" : "Krasnoyarsk Krai",
            "Ярославская область" : "Yaroslavl oblast",
            "Удмуртия" : "Udmurtia",
            "Алтайский край" : "Altai Krai",
            "Иркутская область" : "Irkutsk oblast",
            "Ставропольский край" : "Stavropol Krai",
            "Тульская область" : "Tula oblast"}

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test, len_train, len_test, kf):
    oof_train = np.zeros((len_train,))
    oof_test = np.zeros((len_test,))
    oof_test_skf = np.empty((NFOLDS, len_test))

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
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"

def calcTimeDelta(df, frm, to, predictors):
    changed = False
    nextgrp = [
            #(['region', 'city', 'parent_category_name', 'category_name'], 1, 1),
            #(['lat_lon_hdbscan_cluster_20_03', 'city', 'parent_category_name', 'category_name'], 1, 1),
            (['parent_category_name', 'category_name'], 1, 1),
            (['user_id'], 1, 1),
            #(['user_type'], 1, 1),
            #(['user_id', 'user_type'], 1, 1),
            ]
    
    if not parse_att.checkTimeFeature(df, nextgrp, predictors):
        train_index = df.index
        
        if not debug:
            train_active = pd.read_csv('../input/train_active.csv', index_col = "item_id", parse_dates = ["activation_date"])
            test_active = pd.read_csv('../input/test_active.csv', index_col = "item_id", parse_dates = ["activation_date"])
        else:
            train_active = pd.read_csv('../input/train_active.csv', nrows = 10000, index_col = "item_id", parse_dates = ["activation_date"])
            test_active = pd.read_csv('../input/test_active.csv', nrows = 10000, index_col = "item_id", parse_dates = ["activation_date"])
            
        train_act_index = train_active.index
        test_act_index = test_active.index
        
        print('Train_active shape: {} Rows, {} Columns'.format(*train_active.shape))
        print('Test_active shape: {} Rows, {} Columns'.format(*test_active.shape))
        
        train_active['activation_date'].fillna(train_active.activation_date.min(), inplace=True)
        test_active['activation_date'].fillna(test_active.activation_date.min(), inplace=True)
        
        df = pd.concat([df, train_active, test_active],axis=0)
        del train_active, test_active
        
        changed |= parse_att.calcTimeAtt(df, predictors, nextgrp, frm, to)
        
        df = df.loc[train_index]
    else:
        print('time delta features existed')
        
    return df, changed

def calcImgAtt(df, predictor):
    changed = False
    
    print('load img att dict')
    dict_att = load(dict_att_path)
    def getImageAtt(key, att):
        if key in dict_att[att]:
            return dict_att[att][key]
        else:
            return 0
    
    print('image attrs')
    #img_atts = ['whratio', 'area', 'laplacian', 'colorfull', 'brightness', 'median', 'rms', 'stddev']
    img_atts = ['whratio', 'laplacian', 'colorfull', 'brightness', 'median', 'rms', 'stddev',
                'resnet_conf', 'xception_conf', 'inception_conf']
    for att in tqdm(img_atts):
        if att in df.columns:
            print('{} exist'.format(att))
            continue
        else:
            df[att] = 0
            df[att] = df['image'].map(lambda x: getImageAtt(x, att)).astype(np.float32)
            changed = True
            
    predictor += img_atts
    
    return df, changed

def preparBaseData(y, df, predictors, len_train, len_test, frm, to, tot_filename):
    changed = False
    
    """
    geo_cols = ['latitude', 'longitude', 
                'lat_lon_hdbscan_cluster_05_03', 'lat_lon_hdbscan_cluster_10_03', 
                'lat_lon_hdbscan_cluster_20_03']
    """
    geo_cols = ['latitude', 'longitude']
    if 'longitude' not in df.columns:
        geo = pd.read_csv('../input/avito_region_city_features.csv')[geo_cols + ['region', 'city']]
        df = df.reset_index().merge(geo, how='left', on=["region", "city"]).set_index('item_id')
        changed = True
    predictors += geo_cols
    
    if 'reg_dense' not in df.columns:
        regional = pd.read_csv('../input/regional.csv', index_col=0)
        regional.index = regional.index.str.lower()
        
        df['region'] = df['region'].apply(lambda x : region_map[x])
        df['region'] = df['region'].str.lower()
        df["reg_dense"] = df['region'].apply(lambda x: regional.loc[x,"Density_of_region(km2)"])
        df["rural"] = df['region'].apply(lambda x: regional.loc[x,"Rural_%"])
        df["reg_Time_zone"] = df['region'].apply(lambda x: regional.loc[x,"Time_zone"])
        df["reg_Population"] = df['region'].apply(lambda x: regional.loc[x,"Total_population"])
        df["reg_Urban"] = df['region'].apply(lambda x: regional.loc[x,"Urban%"])
        changed = True
    predictors += ['reg_dense', 'rural', 'reg_Time_zone', 'reg_Population', 'reg_Urban']
    
    if 'avg_days_up_user' not in df.columns:
        train_features = pd.read_csv('../input/aggregated_features.csv')
        df = df.reset_index().merge(train_features, on = ['user_id'], how = 'left').set_index('item_id')
        df['avg_days_up_user'].fillna(0, inplace = True)
        df['avg_times_up_user'].fillna(0, inplace = True)
        df['n_user_items'].fillna(0, inplace = True)
    predictors += ['avg_days_up_user', 'avg_times_up_user', 'n_user_items']
        
    
    #df, timechanged = calcTimeDelta(df, frm, to, predictors)
    #changed |= timechanged
        
    gc.collect()
    print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
    
    if 'population' not in df.columns:
        print('merge population')
        population = pd.read_csv('../input/city_population.csv')
        df = df.reset_index().merge(population, how='left', on='city').set_index('item_id')
        changed = True
    predictors.append('population')
    
    print("Feature Engineering")
    df["price"] = np.log(df["price"]+0.001)
    df["price"].fillna(-999, inplace=True)
    df["image_top_1"].fillna(-999, inplace=True)
    df["population"].fillna(df["population"].min(), inplace=True)
    df["population"] = np.log(df["population"]+0.001)
    df['reg_Population'] = np.log(df["reg_Population"]+0.001)
    df['reg_dense'] = np.log(df["reg_dense"]+0.001)
    
    
    if 'activation_date' in df.columns:
        df.loc[df['activation_date']>'2017-04-18', 'activation_date'] = np.datetime64('2017-04-18')
        
        print("\nCreate Time Variables")
        dt = df['activation_date'].dt
        df["Weekday"] = dt.weekday.astype(np.uint8)
        #df["Weekd of Year"] = dt.week.astype(np.uint8)
        #df["DayofMonth"] = dt.day.astype(np.uint8)
        #df["DayofYear"] = dt.dayofyear.astype(np.uint16)
        #df["Month"] = dt.month.astype(np.uint8)
    
        del(dt)
        gc.collect() 
    
    #predictors += ["Weekday", "Weekd of Year", "DayofMonth", "Month", "price", "item_seq_number"]
    predictors += ["Weekday", "price", "item_seq_number"]
    
    if 'whratio' not in df.columns:
        df_imgatt = pd.read_csv('../input/df_imgatt.csv')
        df = df.reset_index().merge(df_imgatt, how='left', on=["image"]).set_index('item_id')
        changed = True
    
    img_atts = ['whratio', 'laplacian', 'colorfull', 'brightness', 'median', 'rms', 'stddev','resnet_conf', 'xception_conf', 'inception_conf']
    predictors += img_atts
    #df, imgchanged = calcImgAtt(df, predictors)
    #changed |= imgchanged
    
    # Create Validation Index and Remove Dead Variables
    #training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
    #validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index

    print("\nEncode Variables")
    categorical = ["user_id","region","city","parent_category_name",
                   "category_name","user_type","image_top_1",
                   "param_1","param_2","param_3",
                   'reg_Time_zone', ]
    predictors += categorical
    print("Encoding :",categorical)
    
    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in tqdm(categorical):
        df[col].fillna('Unknown')
        df[col] = lbl.fit_transform(df[col].astype(str))
        if col == 'user_id':
            df[col] = df[col].astype(np.uint32)
        else:
            df[col] = df[col].astype(np.uint16)
        #print('max {} {}'.format(col, df[col].max()))
        
    # Feature Engineering 
    
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    count_digit = lambda s : sum(c.isdigit() for c in s)
    count_num = lambda s : sum(c.isnumeric() for c in s.split())
    
    
    # Meta Text Features
    df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    textfeats = ["description", "title"]
    for cols in tqdm(textfeats):
        df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        #df[cols] = df[cols].apply(lambda x: cleanName(x))
        
        att_name = cols + '_num_chars'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = df[cols].apply(len).astype(np.uint16) # Count number of Characters
            changed |= True
            
        att_name = cols + '_num_words'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = df[cols].apply(lambda comment: len(comment.split())).astype(np.uint16) # Count number of Words
            changed |= True
            
        att_name = cols + '_num_unique_words'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = df[cols].apply(lambda comment: len(set(w for w in comment.split()))).astype(np.uint16)
            changed |= True
            
        att_name = cols + '_words_vs_unique'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = (df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100).astype(np.float32) # Count Unique Words
            changed |= True
            
        att_name = cols + '_punctuation'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = df[cols].apply(count, args=(string.punctuation,)).astype(np.uint16)
            changed |= True
           
        att_name = cols + '_digit'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = df[cols].apply(count_digit).astype(np.uint16)
            changed |= True
            
        att_name = cols + '_num'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = df[cols].apply(count_num).astype(np.uint16)
            changed |= True
            
        att_name = cols + '_num_letters'
        predictors.append(att_name)
        if att_name not in df.columns:
            df[att_name] = df[cols].apply(lambda comment: len(comment)).astype(np.uint16)
            changed |= True

    #df['description_num_letters'] = df['description_num_letters'] + 1
    #df['description_num_words'] = df['description_num_words'] + 1
    df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']
    df['desc_num_ratio'] = df['description_num']/df['description_num_words']
    predictors += ['title_desc_len_ratio', 'desc_num_ratio']
    
    df = parse_att.checkDrop_Bulk(df, ["activation_date", "image"])
    
    feature_list = [
            (['city', 'category_name', 'param_1'], ['count', 'nunique']),          
            (['category_name', 'param_1', 'price'], ['count', 'zscore']),           
            (['user_id', 'price'], ['count']),
            (['user_id', 'category_name', 'param_1', 'price'], ['count']),
            (['city', 'category_name', 'param_1', 'price'], ['count', 'zscore']),

            (['category_name', 'param_1', 'param_2', 'description_num_chars'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'description_num_words'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'description_num_unique_words'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'description_words_vs_unique'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'description_punctuation'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'description_digit'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'description_num'], ['zscore']),
            
            
            (['category_name', 'param_1', 'param_2', 'title_num_chars'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'title_num_words'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'title_num_unique_words'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'title_words_vs_unique'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'title_punctuation'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'title_digit'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'title_num'], ['zscore']),
        
            (['category_name', 'param_1', 'param_2', 'title_desc_len_ratio'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'desc_num_ratio'], ['zscore']),
            
            (['category_name', 'param_1', 'param_2', 'whratio'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'laplacian'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'colorfull'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'brightness'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'median'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'rms'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'stddev'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'resnet_conf'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'xception_conf'], ['zscore']),
            (['category_name', 'param_1', 'param_2', 'inception_conf'], ['zscore']),
            ]
    
    
    for (selcol, how) in tqdm(feature_list):
        print('{} {}'.format(selcol, how))
        df, sub_changed = parse_att.calcGroupFeatureBulk(df, selcol, how, frm, to, predictors)
        changed |= sub_changed
        
    for col in df.columns:
        if 'zscore' in col:
            df[col].fillna(0, inplace=True)
            df[col].replace(np.Inf, 0, inplace=True)
            df[col].replace(-np.Inf, 0, inplace=True)
            df[col][df[col] < -4] = -4
            df[col][df[col] > 4] = 4
        
    df = kaggle_util.reduce_mem_usage(df)

    if not changed:
        print('df not changed')
    else:
        print('df changed, save...')
        #df.reset_index().to_feather(tot_filename)
        df.reset_index().to_csv(tot_filename)
    
    old_num = len(predictors)
    predictors = list(set(predictors))
    print('unique feature num from [{}] to [{}]'.format(old_num, len(predictors)))
    return y, df, predictors, len_train, categorical, textfeats

def preparTotalData(y, df, predictors, len_train, len_test, frm, to, tot_filename):

    y, df, predictors, len_train, categorical, textfeats = preparBaseData(y, df, predictors, len_train, len_test, frm, to, tot_filename)
        
    
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
    vectorizer = FeatureUnion([
            ('description',TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=17000,
                **tfidf_para,
                preprocessor=get_col('description'))),
            
            ('title',TfidfVectorizer(
                ngram_range=(1, 2),
                **tfidf_para,
                #max_features=7000,
                preprocessor=get_col('title')))
        ])   
        
    start_vect=time.time()
    #vectorizer.fit(df.loc[traindex,:].to_dict('records'))
    vectorizer.fit(df[:len_train].to_dict('records'))
    ready_df = vectorizer.transform(df.to_dict('records'))
    tfvocab = vectorizer.get_feature_names()
    print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))
    
    # Drop Text Cols
    df.drop(textfeats, axis=1,inplace=True)
    
 
    #from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    kf = KFold(len_train, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                    'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
    ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
    ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:len_train], y, ready_df[len_train:],
                                              len_train, len_test, kf)
    #rms = sqrt(mean_squared_error(y, ridge_oof_train))
    ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
    df['ridge_preds'] = ridge_preds
    predictors.append('ridge_preds')

    df = kaggle_util.reduce_mem_usage(df)
    return y, df, ready_df, tfvocab, predictors, len_train, categorical

def main(frm, to):
    
    testing = pd.read_csv('../input/test.csv', skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
    testdex = testing.index
    len_test = len(testing)
    
    tot_filename = '/media/extend/cache/total_{}_{}.csv'.format(frm, to)
    tot_yname = '/media/extend/cache/total_y_{}_{}.csv'.format(frm, to)
    if os.path.exists(tot_filename) and os.path.exists(tot_yname):
        print('load from feather')
        #df = pd.read_feather(tot_filename).set_index("item_id")
        #y = pd.read_feather(tot_yname).set_index("item_id").deal_probability.copy()
        df = pd.read_csv(tot_filename).set_index("item_id")
        y = pd.read_csv(tot_yname).set_index("item_id").deal_probability.copy()
        
        len_train = to - frm
    else:
        training = pd.read_csv('../input/train.csv', skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
        len_train = len(training)
        
        y = training.deal_probability.copy()
        training.drop("deal_probability",axis=1, inplace=True)
        #y.reset_index().to_feather(tot_yname)
        y.reset_index().to_csv(tot_yname)
        
        print('Train shape: {} Rows, {} Columns'.format(*training.shape))
        print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
        
        df = pd.concat([training, testing], axis=0)
        del training, testing
        
    predictors = []
    y, df, ready_df, tfvocab, predictors, len_train, categorical =  \
        preparTotalData(y, df, predictors, len_train, len_test, frm, to, tot_filename)

    none_categorical = [x for x in df.columns if x not in categorical]
    
    df = df[predictors]
    print(df.info())
    
    print("Modeling Stage")
    X = hstack([csr_matrix(df[:len_train].values),ready_df[0:len_train]]) # Sparse Matrix
    testing = hstack([csr_matrix(df[len_train:].values),ready_df[len_train:]])
    tfvocab = df.columns.tolist() + tfvocab
    for shape in [X,testing]:
        print("{} Rows and {} Cols".format(*shape.shape))
    print("Feature Names Length: ",len(tfvocab))
    del df
    gc.collect();
    
    print("\nModeling Stage")
    
    # Training and Validation Set
    """
    Using Randomized train/valid split doesn't seem to generalize LB score, so I will try time cutoff
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=5)
    """
    total_len = X.shape[0]
    train_len = int(total_len * 0.9)
    X = X.tocsr()
    X_train = X[:train_len]
    X_valid = X[train_len:]
    y_train = y[:train_len]
    y_valid = y[train_len:]
    """ 
    print("Light Gradient Boosting Regressor")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        #'max_depth': 15,
        'num_leaves': 270,# 37,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.85,
        # 'bagging_freq': 5,
        'learning_rate': 0.018,
        'nthread': 6,
        'verbose': 0,
        #'device':'gpu',
        #'gpu_platform_id':0,
        #'gpu_device_id':0
    }  

    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    
    # Go Go Go
    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=26000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # Feature Importance Plot
    #f, ax = plt.subplots(figsize=[7,10])
    #lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
    #plt.title("Light GBM Feature Importance")
    #plt.savefig('feature_import.png', bbox_inches='tight')
    
    print("Model Evaluation Stage")
    rmse = np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid)))
    print('RMSE:', rmse)
    
    str_now = datetime.now().strftime("%m-%d-%H-%M")
    if not debug:
        lgb_clf.save_model('../model/model_{}.txt'.format(str_now), lgb_clf.best_iteration)
    else:
        lgb_clf.save_model('../model/model_debug.txt', lgb_clf.best_iteration)
   
    #lgb_clf = lgb.Booster(model_file='../model/model_05-13-21-50.txt')

    lgpred = lgb_clf.predict(testing, num_iteration = lgb_clf.best_iteration)
    lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
    lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
    #lgsub.to_csv("lgsub.csv",index=True,header=True)
    
    if not debug:
        kaggle_util.save_result(lgsub, '../result/dense_feature_{}.csv'.format(str_now), competition = 'avito-demand-prediction', send = True, index = True)
    print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
    print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

    
def get_crossvalid_data(frm, to):
    #test_path = '../input/test.csv'
    #train_path = '../input/train.csv'
    
    test_path = '../input/imgtop_test.csv'
    train_path = '../input/imgtop_train.csv'
    
    testing = pd.read_csv(test_path, skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
    testdex = testing.index
    len_test = len(testing)
    
    tot_filename = '/media/extend/cache/total_{}_{}.csv'.format(frm, to)
    tot_yname = '/media/extend/cache/total_y_{}_{}.csv'.format(frm, to)
    if os.path.exists(tot_filename) and os.path.exists(tot_yname):
        print('load from feather')
        #df = pd.read_feather(tot_filename).set_index("item_id")
        #y = pd.read_feather(tot_yname).set_index("item_id").deal_probability.copy()
        df = pd.read_csv(tot_filename).set_index("item_id")
        y = pd.read_csv(tot_yname).set_index("item_id").deal_probability.copy()
        
        len_train = to - frm
    else:
        training = pd.read_csv(train_path, skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
        len_train = len(training)
        
        y = training.deal_probability.copy()
        training.drop("deal_probability",axis=1, inplace=True)
        #y.reset_index().to_feather(tot_yname)
        y.reset_index().to_csv(tot_yname)
        
        print('Train shape: {} Rows, {} Columns'.format(*training.shape))
        print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
        
        df = pd.concat([training, testing], axis=0)
        del training, testing
        
    predictors = []
    y, df, ready_df, tfvocab, predictors, len_train, categorical =  \
        preparTotalData(y, df, predictors, len_train, len_test, frm, to, tot_filename)

    #none_categorical = [x for x in df.columns if x not in categorical]
    
    df = df[predictors]
    df = kaggle_util.reduce_mem_usage(df)
    print(df.info())
    
    tfvocab = df.columns.tolist() + tfvocab
    testing = hstack([csr_matrix(df[len_train:].values),ready_df[len_train:]])
    
    return df, y, testing, ready_df, tfvocab, predictors, len_train, categorical, tfvocab, testdex

def main_crossvalid(frm, to):
    nfold = 5
    df, y, testing, ready_df, tfvocab, predictors, len_train, categorical, tfvocab, testdex = get_crossvalid_data(frm, to)
    
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 270,# 37,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.65,
        'bagging_freq': 2,
        'learning_rate': 0.016,
        #'max_depth' : 8,
        #'min_split_gain' : 0.0222415,
        #'min_child_weight' : 20,
        'nthread': 5,
        'verbose': 0,
        #'reg_alpha' : 0.041545473,
        #'reg_lambda' : 0.0735294,
        'drop_rate': 0.08
        } 
    
    skf = StratifiedKFold(y, n_folds=nfold)
    
    for i, (train_split, val_split) in enumerate(skf):
        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=5)
        print(train_split)
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
        
        modelstart = time.time()
        lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=26000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        print("Model Evaluation Stage")
        rmse = np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid)))
        print('RMSE:', rmse)
        
        f, ax = plt.subplots(figsize=[7,10])
        lgb.plot_importance(lgb_clf, max_num_features=100, ax=ax)
        plt.title("Light GBM Feature Importance")
        plt.savefig('feature_import.png', bbox_inches='tight')
        
        str_now = datetime.now().strftime("%m-%d-%H-%M")
        if not debug:
            lgb_clf.save_model('../model/model_{}.txt'.format(i), lgb_clf.best_iteration)
        else:
            lgb_clf.save_model('../model/model_debug_{}.txt'.format(i), lgb_clf.best_iteration)
       
        lgpred = lgb_clf.predict(testing, num_iteration = lgb_clf.best_iteration)
        lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
        lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
        
        subfile = '../result/dense_feature_{}.csv'.format(i)
        if debug:
            subfile = '../result/dense_feature_debug{}.csv'.format(i)
        kaggle_util.save_result(lgsub, subfile, competition = 'avito-demand-prediction', send = False, index = True)
        
    result_list = []
    for i in range(nfold):
        subfile = '../result/dense_feature_{}.csv'.format(i)
        if debug:
            subfile = '../result/dense_feature_debug{}.csv'.format(i)
        result_list.append((subfile, 1 / nfold))
        
    kaggle_util.ensemble(result_list, False, 
                         competition = 'avito-demand-prediction', 
                         score_col = 'deal_probability',
                         prefix = 'lgb_avg')
    
def main_crossvalid_xgboost(frm, to):
    import xgboost as xgb
    
    nfold = 5
    df, y, testing, ready_df, tfvocab, predictors, len_train, categorical, tfvocab, testdex = get_crossvalid_data(frm, to)

    cat_features = []
    cols = list(df.columns)
    for col in categorical:
        cat_features.append(cols.index(col))
    
    #lgtest = xgb.DMatrix(testing.toarray())
    #del testing
    #gc.collect()
    
    skf = StratifiedKFold(y, n_folds=nfold)
    
    for i, (train_split, val_split) in enumerate(skf):
        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=5)
        print(train_split)
        X_train = hstack([csr_matrix(df.iloc[train_split].values),ready_df[train_split]])
        X_valid = hstack([csr_matrix(df.iloc[val_split].values),ready_df[val_split]]) # Sparse Matrix 
        y_train = y[train_split] 
        y_valid = y[val_split]
        
        #lgtrain = xgb.DMatrix(X_train.toarray(), label = y_train)
        #lgvalid = xgb.DMatrix(X_valid.toarray(), label = y_valid)
        
        #del X_train, X_valid, y_train
        #gc.collect()
        
        modelstart = time.time()
        
        bst = xgb.XGBRegressor(n_estimators=400, 
                                 booster = 'gbtree',
                                 learning_rate=0.016, 
                                 gamma=0, 
                                 subsample=0.75, 
                                 colsample_bylevel=0.5, 
                                 max_depth=16,
                                 nthread = 6)
                                 
        bst.fit(X_train, y_train,
                eval_set = [(X_train, y_train), (X_valid, y_valid)], 
                verbose = False,
                early_stopping_rounds = 100)

        
        print("Model Evaluation Stage")
        ypre = bst.predict(X_valid)
        rmse = np.sqrt(metrics.mean_squared_error(y_valid, ypre))
        print('RMSE:', rmse)
        """
        f, ax = plt.subplots(figsize=[7,10])
        xgb.plot_importance(bst, ax=ax, max_num_features = 50)
        plt.title("Light GBM Feature Importance")
        plt.savefig('xgb_feature_import.png', bbox_inches='tight')
        """
       
        lgpred = bst.predict(testing)
        lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
        lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
        
        subfile = '../result/xgb_dense_feature_{}.csv'.format(i)
        if debug:
            subfile = '../result/xgb_dense_feature_debug{}.csv'.format(i)
        kaggle_util.save_result(lgsub, subfile, competition = 'avito-demand-prediction', send = False, index = True)
        
    result_list = []
    for i in range(nfold):
        subfile = '../result/xgb_dense_feature_{}.csv'.format(i)
        if debug:
            subfile = '../result/xgb_dense_feature_debug{}.csv'.format(i)
        result_list.append((subfile, 1 / nfold))
        
    kaggle_util.ensemble(result_list, not debug, 
                         competition = 'avito-demand-prediction', 
                         score_col = 'deal_probability',
                         prefix = 'xgb_avg')
    
def train_round(df_train, y, len_train, categorical):
    X = df_train[:len_train].values
    tfvocab_train = df_train.columns.tolist()

    del df_train
    gc.collect();
    
    total_len = X.shape[0]
    train_len = int(total_len * 0.8)
    #X = X.tocsr()
    X_train = X[:train_len]
    X_valid = X[train_len:]
    y_train = y[:train_len]
    y_valid = y[train_len:]
        
    print("Light Gradient Boosting Regressor")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 300,# 37,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.85,
        'learning_rate': 0.019,
        'nthread': 12,
        'verbose': 0,
        #'device':'gpu',
        #'gpu_platform_id':0,
        #'gpu_device_id':0
    }  

    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=tfvocab_train,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=tfvocab_train,
                    categorical_feature = categorical)
    
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=26000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=500
    )
    
    rmse = np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid)))
    return lgb_clf, rmse

def prepare_featuredata(frm, to):
    predictors = []
    
    testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
    len_test = len(testing)
    
    tot_filename = '/media/extend/cache/patial_{}_{}'.format(frm, to)
    tot_yname = '/media/extend/cache/patial_y_{}_{}'.format(frm, to)
    if os.path.exists(tot_filename) and os.path.exists(tot_yname):
        print('load from feather')
        df = pd.read_feather(tot_filename).set_index("item_id")
        y = pd.read_feather(tot_yname).set_index("item_id").deal_probability.copy()
        
        len_train = to - frm
    else:
        #tot_filename = '/media/extend/cache/total_{}_{}'.format(frm, to)
        training = pd.read_csv('../input/train.csv', skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
        len_train = len(training)
        
        y = training.deal_probability.copy()
        training.drop("deal_probability",axis=1, inplace=True)
        y.reset_index().to_feather(tot_yname)
        
        
        print('Train shape: {} Rows, {} Columns'.format(*training.shape))
        print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
        
        df = pd.concat([training, testing], axis=0)
        del training, testing
        
    y, df, predictors, len_train, categorical, textfeats = \
    preparBaseData(y, df, predictors, len_train, len_test, frm, to, tot_filename)
    
    df.drop('description', axis=1, inplace = True)
    df.drop('title', axis=1, inplace = True)
    
    df = df[predictors]
    
    return df, y, categorical, len_train
    
def feature_engineer(frm, to):
    df, y, categorical, len_train = prepare_featuredata(frm, to)
        
    import itertools
    
    none_categorical = [x for x in df.columns if x not in categorical]
    max_len = len(none_categorical)
    
    if debug:
        noncat_save_path = '../input/best_dict_debug.pkl'
    else:
        noncat_save_path = '../input/best_dict.pkl'
    best_dict = {'rmse':1.0, 'noncat':[], 'num_col':max_len, 'iter_idx':0,
                 'best_num_col' : 0, 'best_iter_idx' : 0}
    
    if os.path.exists(noncat_save_path):
        best_dict = kaggle_util.load(noncat_save_path)
        if 'best_num_col' not in best_dict:
            best_dict['best_num_col'] = 0
        if 'best_iter_idx' not in best_dict:
            best_dict['best_iter_idx'] = 0
            
        print('\nload best score [{}] best_num_col [{}] best_iter [{}]\nnum_col [{}] iter [{}]\n'.format(best_dict['rmse'], 
            best_dict['best_num_col'], best_dict['best_iter_idx'], best_dict['num_col'], best_dict['iter_idx']))
    
    for num in tqdm(range(best_dict['num_col'], 1, -1)):
        
        iter_idx = 0
        for sub_noncat in tqdm(itertools.combinations(none_categorical, num)):
            if num == best_dict['num_col'] and iter_idx < best_dict['iter_idx']:
                #print('skip num_col [{}] iter [{}]'.format(num, iter_idx))
                iter_idx += 1
                continue
            
            print('\nnum_col [{}] iter [{}]\n'.format(num, iter_idx))
            iter_idx += 1
            sub_preds = categorical + list(sub_noncat)
            df_train = df[sub_preds]
            
            lgb_clf, rmse = train_round(df_train, y, len_train, categorical)
            
            if rmse < best_dict['rmse']:
                print('** best RMSE [{}] with {} features'.format(rmse, num))
                
                best_dict['rmse'] = rmse
                best_dict['noncat'] = list(sub_noncat)
                best_dict['best_num_col'] = num
                best_dict['best_iter_idx'] = iter_idx
                
            best_dict['num_col'] = num
            best_dict['iter_idx'] = iter_idx
            kaggle_util.save(best_dict, noncat_save_path)
                
def feature_engineer_minus(frm, to):
    
    
    df, y, categorical, len_train = prepare_featuredata(frm, to)
    len_data = to - frm
    
    none_categorical = [x for x in df.columns if x not in categorical]
    max_len = len(none_categorical)
    
    if debug:
        noncat_save_path = '../input/best_dict_minus_debug_{}.pkl'.format(len_data)
    else:
        noncat_save_path = '../input/best_dict_minus_{}.pkl'.format(len_data)
    best_dict = {'best_rmse':1.0, 'base_rmse':1.0, 
                 'num_col':max_len, 'iter_idx':0,
                 'best_minus': None,
                 'minus_improve':[], 
                 'wait_validate':none_categorical,
                 'next_wait_validate':[]
                 }
    
    if debug:
        best_dict['wait_validate'] = none_categorical[:10]
    
    if os.path.exists(noncat_save_path):
        best_dict = kaggle_util.load(noncat_save_path)
            
        print('\nload best score [{}] num_col [{}] iter [{}]\n'.format(
                best_dict['best_rmse'], best_dict['num_col'], 
                best_dict['iter_idx']))
        
    else:
        lgb_clf, rmse = train_round(df, y, len_train, categorical)
        best_dict['best_rmse'] = rmse
        best_dict['base_rmse'] = rmse
    
    valid_none_categorical = [x for x in none_categorical if x not in best_dict['minus_improve']]
    
    found = True
    while found:
        found = False
        
        next_wait_validate = best_dict['next_wait_validate'][:]
        wait_validate = best_dict['wait_validate'][:]
        
        print('\n**********************')
        print('NEW ROUND')
        print('base_rmse: {}'.format(best_dict['base_rmse']))
        print('len_wait_validate: {}'.format(len(wait_validate)))
        print('len_next_wait_validate: {}'.format(len(next_wait_validate)))
        print('**********************\n')
        
        iter_idx = 0
        num = len(wait_validate)
        for minus_cat in tqdm(wait_validate):
            if num == best_dict['num_col'] and iter_idx < best_dict['iter_idx']:
                iter_idx += 1
                continue
            iter_idx += 1
            
            sub_noncat = valid_none_categorical[:]
            sub_noncat.remove(minus_cat)
        
            print('\nnum_col [{}] iter [{}]\n'.format(num, iter_idx))
            sub_preds = categorical + list(sub_noncat)
            df_train = df[sub_preds]
            
            lgb_clf, rmse = train_round(df_train, y, len_train, categorical)
            
            if rmse <= best_dict['base_rmse']:
                found = True
                
                print('improve RMSE [{}] with {} features'.format(rmse, num))
                
                if rmse <= best_dict['best_rmse']:
                    print('** best RMSE [{}] with {} features'.format(rmse, num))
                    
                    if best_dict['best_minus'] is not None:
                        next_wait_validate.append(best_dict['best_minus'])
                    
                    best_dict['best_rmse'] = rmse
                    best_dict['best_minus'] = minus_cat
                else:
                    next_wait_validate.append(minus_cat)
                
                
            best_dict['num_col'] = num
            best_dict['iter_idx'] = iter_idx
            #best_dict['wait_validate'].remove(minus_cat)
            best_dict['next_wait_validate'] = next_wait_validate
            kaggle_util.save(best_dict, noncat_save_path)
        
        if best_dict['best_minus'] is not None:
            best_dict['minus_improve'].append(best_dict['best_minus'])
        
            #best_dict['wait_validate'] = best_dict['next_wait_validate']
            best_dict['wait_validate'].remove(best_dict['best_minus'])
            
            print('**improve from {} to {}'.format(best_dict['base_rmse'], best_dict['best_rmse']))
        
        if found:
            best_dict['iter_idx'] = 0
        best_dict['best_minus'] = None
        best_dict['next_wait_validate'] = []
        best_dict['base_rmse'] = best_dict['best_rmse']
        
        kaggle_util.save(best_dict, noncat_save_path)
    
if __name__ == "__main__":
    print("\nData Load Stage")
    
    frm = 0
    to = 1503424
    if debug:    
        frm = 0
        to = 10000
    #main(frm, to)
    main_crossvalid(frm, to)
    #main_crossvalid_xgboost(frm, to)
    
    frm = 1503424 - 500000
    to = 1503424
    if debug:    
        frm = 0
        to = 11000
    
    #feature_engineer_minus(frm, to)
    """
    result_list = [
            ('dense_feature_mix_0.2205.csv', 0.4),
            ('xgb_tfidf_0.2229.csv', 0.3),
            ('rnn_0.2209.csv', 0.3),
            ]

    kaggle_util.ensemble(result_list, True, 
                         competition = 'avito-demand-prediction', 
                         score_col = 'deal_probability')
    """
    """
    result_list = []
    for i in range(1, 6):
        subfile = '../result/xgb_tfidf_{}.csv'.format(i)
        result_list.append((subfile, 1 / 10))
        
    kaggle_util.ensemble(result_list, not debug, 
                         competition = 'avito-demand-prediction', 
                         score_col = 'deal_probability')
    """
