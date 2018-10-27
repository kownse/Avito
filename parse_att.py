#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:07:02 2018

@author: kownse
"""
import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm

#from train_lightgbm import debug

#debug = 1
def df_add_counts(df, cols, name = ''):
    if len(name) == 0:
        name = "_".join(cols)+'_count'
    #print(name)
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
									 return_inverse=True, return_counts=True)
    df[name] = counts[unqtags].astype('uint16')
    del(unqtags)
    del(unq)
    del(counts)
    del(arr_slice)
    gc.collect()
    
def checkDrop(df, col):
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
    return df
        
def checkDrop_Bulk(df, cols):
    for col in cols:
        df = checkDrop(df, col)
    return df
    
def shrink_gp(gp, selcols):
    for col in selcols:
        gp[col] = gp[col].astype('uint32')
    return gp

log_group = np.log(100000)
def rate_calculation(x):
    """Calculate the attributed rate. Scale by confidence"""
    rate = x.sum() / float(x.count())
    conf = np.min([1, np.log(x.count()) / log_group])
    return rate * conf
    
def calcGroupFeature(train_df, selcols, how, frm, to, predictors, na=0):
    #assert(how in ['count', 'mean', 'var', 'skew', 'nunique', 'cumcount', 
    #               'confidence', 'feature_count', 'zscore'])
        
    att_name = '_'.join(selcols + [how])
    if att_name in train_df.columns:
        #print('{} already in train_df'.format(att_name))
        
        if att_name not in predictors:
            predictors.append(att_name)
        return train_df, False
    
    print('group feature: ' + att_name)
    train_df = train_df.reset_index()
    
    by_cols = selcols[0:len(selcols)-1]
    tar_col = selcols[len(selcols)-1]
    #filename = '../cache/{}[{},{}].csv'.format(att_name, frm,to)
    feather_path = '/media/extend/cache/{}[{},{}].feather'.format(att_name, frm,to)
    gp_exist = True
    if os.path.exists(feather_path):
        print('load from file')
        if how=='cumcount': 
            #gp=pd.read_csv(filename,header=None)
            gp = pd.read_feather(feather_path)
            train_df[att_name] = gp['cumcount']
            train_df[att_name] = train_df[att_name].astype('uint16')
        elif how == 'zscore':
            gp = pd.read_feather(feather_path).set_index('index')
            train_df = train_df.merge(gp, on=by_cols, how='left')
            train_df[att_name] = (train_df[tar_col] - train_df['mean']) / train_df['var']
            train_df = checkDrop_Bulk(train_df, ['mean','var'])
        else: 
            #gp=pd.read_csv(filename)
            gp = pd.read_feather(feather_path).drop('index', axis=1)
            
            train_df = train_df.merge(gp, on=by_cols, how='left')
    else:
        print('calculate from scratch: ' + att_name)
        
        if how=='count':
            
            gp = train_df[selcols].groupby(by=by_cols)[tar_col].count().reset_index().\
                rename(index=str, columns={tar_col: att_name})
            train_df = train_df.merge(gp, on=by_cols, how='left')
            
            # faster, but strange
            #df_add_counts(train_df, by_cols, att_name)
            gp_exist = False
        elif how=='mean':
            gp = train_df[selcols].groupby(by=by_cols)[tar_col].mean().reset_index().\
                rename(index=str, columns={tar_col: att_name})
            
            gp = shrink_gp(gp, by_cols)
            gp[att_name] = gp[att_name].astype('float32')
            
            train_df = train_df.merge(gp, on=by_cols, how='left')
            train_df[att_name] = train_df[att_name].fillna(0).astype('float32')
            gp = gp.reset_index()
            gp['index'] = gp['index'].astype('uint32')
        elif how=='var':
            gp = train_df[selcols].groupby(by=by_cols)[tar_col].var().reset_index().\
                rename(index=str, columns={tar_col: att_name})
                
            gp = shrink_gp(gp, by_cols)
            gp[att_name] = gp[att_name].astype('float32')
            
            train_df = train_df.merge(gp, on=by_cols, how='left')
            train_df[att_name] = train_df[att_name].fillna(0).astype('float32')
            gp = gp.reset_index()
            gp['index'] = gp['index'].astype('uint32')
        elif how=='zscore':
            
            print(selcols)
            sub_df = train_df[selcols]
            gp = sub_df.groupby(by=by_cols)[tar_col].agg(['mean','std']).reset_index().\
                rename(index=str)
            gp['std'].fillna(0.0001, inplace=True)
            gp = shrink_gp(gp, by_cols)
            
            train_df = train_df.merge(gp, on=by_cols, how='left')
            train_df[att_name] = (train_df[tar_col] - train_df['mean']) / train_df['std']
            train_df = checkDrop_Bulk(train_df, ['mean','std'])
            
            gp = gp.reset_index()
            gp['index'] = gp['index'].astype('uint32')
                
        elif how=='skew':
            gp = train_df[selcols].groupby(by=by_cols)[tar_col].skew().reset_index().\
                rename(index=str, columns={tar_col: att_name})
                
            gp = shrink_gp(gp, by_cols)
            gp[att_name] = gp[att_name].astype('float16')
            
            train_df = train_df.merge(gp, on=by_cols, how='left')
            train_df[att_name] = train_df[att_name].fillna(0).astype('float16')
            
            gp = gp.reset_index()
            gp['index'] = gp['index'].astype('uint32')
        elif how=='nunique':
            gp = train_df[selcols].groupby(by=by_cols)[tar_col].nunique().reset_index().\
                rename(index=str, columns={tar_col: att_name})
                
            gp = shrink_gp(gp, by_cols)
            gp[att_name] = gp[att_name].astype('uint16')
            
            train_df = train_df.merge(gp, on=by_cols, how='left')
            train_df[att_name] = train_df[att_name].fillna(0)
            train_df[att_name] = train_df[att_name].astype('uint16')
            
            gp = gp.reset_index()
            gp['index'] = gp['index'].astype('uint32')
        elif how=='cumcount':
            gp = train_df[selcols].groupby(by=by_cols)[tar_col].cumcount()
            gp = gp.astype('uint16')

            train_df[att_name]=gp.values
            train_df[att_name] = train_df[att_name].astype('uint16')
            
            gp = gp.rename('cumcount').reset_index()
        elif how=='confidence':
            gp = train_df.groupby(by_cols)[tar_col].agg(['count','mean']).reset_index()
            gp[att_name] = gp[['count','mean']].apply(lambda x: x.ix[1] * np.min([1, np.log(x.ix[0])/np.log(10000)]), axis=1)
            gp.drop(['count','mean'], axis=1, inplace=True)
                 
            gp = shrink_gp(gp, by_cols)
            gp[att_name] = gp[att_name].astype('float16')
            
            
            train_df = train_df.merge(gp, on=by_cols, how='left')
            train_df[att_name] = train_df[att_name].astype('float16')
        
        if gp_exist:# and not debug:
             #gp.to_csv(filename,index=False)
             gp.to_feather(feather_path)
             
    if gp_exist:    
        del gp
    gc.collect() 
    train_df[att_name].fillna(na, inplace=True)
    train_df = train_df.set_index('item_id')
    predictors.append(att_name)
    
    return train_df, True

def calcGroupFeatureBulk(train_df, selcols, hows, frm, to, predictors):
    changed = False
    
    if type(hows) == list:
        for how in hows:
            train_df, c = calcGroupFeature(train_df, selcols, how, frm, to, predictors)
            changed |= c
        return train_df, changed
    else:
        return calcGroupFeature(train_df, selcols, hows, frm, to, predictors)

def checkCreateClickCategory(train_df, D, cols):
    if 'category' not in train_df.columns:
        str_col = train_df[cols[0]].astype(str)
        for i in range(1, len(cols)):
            str_col += '_' + train_df[cols[i]].astype(str)
        train_df['category'] = (str_col.apply(hash) % D).astype('uint32')
        
        #train_df['category'] = ((train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
        #    + "_" + train_df['os'].astype(str)).apply(hash) % D).astype('uint32')
    if 'epochtime' not in train_df.columns:
        train_df['epochtime']= train_df['activation_date'].astype(np.int64) // 10 ** 9

def calcNextClick(train_df, predictors, step, frm, to, D, cols = ['ip', 'app', 'device', 'os']):
    if step <= 0:
        return False
    #print('nextClick')
    feature = 'nextClick_{}'.format('_'.join(cols))
    #print(feature)
    filename='/media/extend/cache/{}_{}_{}.npy'.format(feature, frm,to)
    
    
    att_name = name = '{}_{}'.format(feature, 0)
    if att_name in train_df.columns:
        #print('{} already in train_df'.format(att_name))
        
        for i in range(step):
            name = '{}_{}'.format(feature, i)
            if name not in predictors:
                predictors.append(name)
                
        return False
 
    next_clicks= []
    if os.path.exists(filename):
        print('loading from save file')
        next_clicks = np.load(filename)
    else:
        checkCreateClickCategory(train_df, D, cols)
        click_buffer = np.full((D, step), 2**32-1, dtype=np.uint32)
        
        for category, t in tqdm(zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values))):
            next_clicks.append(click_buffer[category] - t)
            click_buffer[category] = np.roll(click_buffer[category], 1)
            click_buffer[category, 0]= t
            
        del(click_buffer)
        next_clicks = np.array(next_clicks)
        #QQ= list(reversed(next_clicks))

        if True:#not debug:
            print('saving')
            np.save(filename, next_clicks)

    
    for i in range(step):
        name = '{}_{}'.format(feature, i)
        col = next_clicks[:, i]
        reverse = col[::-1]
        train_df[name] = reverse
        predictors.append(name)
        
    return True
    
def calcLastClick(train_df, predictors, step, frm, to, D, cols = ['ip', 'app', 'device', 'os']):
    if step <= 0:
        return False
    #print('lastClick')
    feature = 'lastClick_{}'.format('_'.join(cols))
    #print(feature)
    filename='/media/extend/cache/{}_{}_{}.npy'.format(feature, frm,to)
    
    att_name = name = '{}_{}'.format(feature, 0)
    if att_name in train_df.columns:
        #print('{} already in train_df'.format(att_name))
        
        for i in range(step):
            name = '{}_{}'.format(feature, i)
            if name not in predictors:
                predictors.append(name)
        return False
 
    next_clicks= []
    if os.path.exists(filename):
        print('loading from save file')
        next_clicks = np.load(filename)
    else:
        checkCreateClickCategory(train_df, D, cols)
        click_buffer = np.full((D, step), 0, dtype=np.uint32)
        
        for category, t in tqdm(zip(train_df['category'].values, train_df['epochtime'].values)):
            next_clicks.append(t - click_buffer[category])
            click_buffer[category] = np.roll(click_buffer[category], 1)
            click_buffer[category, 0]= t
            
        del(click_buffer)
        next_clicks = np.array(next_clicks)

        if True:#not debug:
            print('saving')
            np.save(filename, next_clicks)

    for i in range(step):
        name = '{}_{}'.format(feature, i)
        train_df[name] = next_clicks[:, i]
        predictors.append(name)
    
    return True

def calcLastAndNextClick(train_df, nextstep, laststep, predictors, frm, to, cols):
    print('calc Last & Next group by {}'.format('_'.join(cols)))
    
    changed = False
    D=2**26
    changed |= calcNextClick(train_df, predictors, nextstep, frm, to, D, cols = cols)
    changed |= calcLastClick(train_df, predictors, laststep, frm, to, D, cols = cols)
    
    checkDrop(train_df, 'category')
    return changed

def calcTimeAtt(train_df, predictors, nextgrp, frm, to):
    changed = False
    for grpcols, nextstep, laststep in nextgrp:
        changed |= calcLastAndNextClick(train_df, nextstep, laststep, predictors, frm, to, cols = grpcols)
    
    checkDrop(train_df, 'click_time')
    checkDrop(train_df, 'epochtime')

    gc.collect()
    return changed

def checkTimeFeature(train_df, grps, predictors):
    
    add_pre = []
    for cols, nextstep, laststep in grps:
        next_feature = 'nextClick_{}'.format('_'.join(cols))
        last_feature = 'lastClick_{}'.format('_'.join(cols))
        
        for i in range(nextstep):
            att_name = '{}_{}'.format(next_feature, i)
            add_pre.append(att_name)
            if att_name not in train_df.columns:
                return False
            
        for i in range(laststep):
            att_name = '{}_{}'.format(last_feature, i)
            add_pre.append(att_name)
            if att_name not in train_df.columns:
                return False
    predictors += add_pre
    return True

def cvsToFeather(selcols, how, frm, to):
    assert(how in ['count', 'mean', 'var', 'skew', 'nunique', 'cumcount', 
                   'confidence', 'feature_count'])
        
    att_name = '_'.join(selcols + [how])
    print('group feature: ' + att_name)

    filename = '../cache/{}[{},{}].csv'.format(att_name, frm,to)
    feather_path = '../cache/{}[{},{}].feather'.format(att_name, frm,to)
    by_cols = selcols[0:len(selcols)-1]

    if os.path.exists(filename) and not os.path.exists(feather_path):
        print('load from file')
        if how=='cumcount': 
            gp=pd.read_csv(filename,header=None)
            gp = gp.astype('uint16')
            gp = gp.rename(columns={0: 'cumcount'}).reset_index()
            gp['index'] = gp['index'].astype('uint32')
        else: 
            gp=pd.read_csv(filename)
            type = 'uint16'
            if how in ['var', 'confidence']:
                type = 'float32'
            gp[att_name] = gp[att_name].astype(type)
            gp = shrink_gp(gp, by_cols)
        gp.to_feather(feather_path)
