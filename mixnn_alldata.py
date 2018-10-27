#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:10:41 2018

@author: kownse
"""

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import parse_att
from attention_with_context import AttentionWithContext

import keras.backend as K
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import \
    BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, GRU, Bidirectional, GlobalMaxPooling1D, Conv1D
from keras.layers import CuDNNGRU
from keras.layers import add, dot
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from itertools import combinations

from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import kaggle_util
from datetime import datetime

from lightgbm_avito import calcTimeDelta, calcImgAtt
from sklearn import preprocessing
import string

DATA_DIR = '../input/'
EMB_PATH = '../input/wiki.ru.vec'
#EMB_PATH = '../input/fasttext-russian-2m/wiki.ru.vec'
#EMB_PATH = '../input/russian-glove/multilingual_embeddings.ru'
target_col = 'deal_probability'
max_features = 50000
maxlen = 100
embed_size = 300

debug = 0

def add_group_size(df, by, y='price'):
    grp = df.groupby(by)[y].size().map(lambda x:np.log1p(x))
    grp = grp.rename('size_'+'_'.join(by)).reset_index()
    df = df.merge(grp, on=by, how='left')
    return df

def add_group_mean(df, by, y='price'):
    grp = df.groupby(by)[y].mean()
    grp = grp.rename('mean_price_'+'_'.join(by)).reset_index()
    df = df.merge(grp, on=by, how='left')
    return df

def get_coefs(word, *arr, tokenizer=None):
    if tokenizer is None:
        return word, np.asarray(arr, dtype='float32')
    else:
        if word not in tokenizer.word_index:
            return None
        else:
            return word, np.asarray(arr, dtype='float32')
        
def fill_rand_norm(embedding_matrix):
    emb_zero_shape = embedding_matrix[embedding_matrix==0].shape
    emb_non_zero_mean = embedding_matrix[embedding_matrix!=0.].mean()
    emb_non_zero_std = embedding_matrix[embedding_matrix!=0.].std()
    embedding_matrix[embedding_matrix==0] = np.random.normal(emb_non_zero_mean, 
                                                             emb_non_zero_std, 
                                                             emb_zero_shape)
    return embedding_matrix

def get_keras_data(df, text):
    X = {}
    for c in df.columns:
        X[c] = df[c].values
    X['text'] = text
    return X

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

"""
MIXED ARCH NN
"""
def get_model(params):
    cats = [Input(shape=[1], name=name) for name in params['cat_feats']]
    nums = [Input(shape=[1], name=name) for name in params['num_feats']]
    emb_fn = lambda name: Embedding(params['emb_cat_max'][name], params['emb_size'])
    embs = []
    for name, cat in zip(params['cat_feats'], cats):
        embs.append(emb_fn(name)(cat))
    
    texts = Input(shape=(params['maxlen'],), name='text')
    text_emb = Embedding(params['nb_words'], 
                         params['word_emb_size'], 
                         weights=[params['embedding_matrix']],
                         name='text_emb')(texts)
    x_text = SpatialDropout1D(params['text_emb_dropout'])(text_emb)
    
    outs = []
    if params['use_rnn']:
        x_rnn = Bidirectional(GRU(params['n_rnn'], return_sequences=True))(x_text)
        avg_pool_rnn = GlobalAveragePooling1D()(x_rnn)
        max_pool_rnn = GlobalMaxPooling1D()(x_rnn)
        outs += [avg_pool_rnn, max_pool_rnn]
    if params['use_cnn']:
        x_cnn = Conv1D(**params['cnn_param'])(x_text)
        avg_pool_cnn = GlobalAveragePooling1D()(x_cnn)
        max_pool_cnn = GlobalMaxPooling1D()(x_cnn)
        outs += [avg_pool_cnn, max_pool_cnn]
    if params['use_att']:
        x_att = AttentionWithContext()(x_text)
        outs += [x_att]
        if params['use_rnn']:
            x_rnn_att = AttentionWithContext()(x_rnn)
            outs += [x_rnn_att]
        if params['use_cnn']:
            x_cnn_att = AttentionWithContext()(x_cnn)
            outs += [x_cnn_att]
    if params['use_fm']:
        first_order = [Flatten()(emb0) for emb0 in embs]
        second_order = []
        for emb1, emb2 in combinations(embs, 2):
            dot_layer = dot([Flatten()(emb1), Flatten()(emb2)], axes=1)
            second_order.append(dot_layer)
        first_order = add(first_order)
        second_order = add(second_order)
        outs += [first_order, second_order]
    if params['use_deep']:
        all_in = [Flatten()(emb) for emb in embs] + nums + [Flatten()(text_emb)]
        x_in = concatenate(all_in)
        for idx, (drop_p, num_dense) in enumerate(zip(params['drop_out'], params['deep_layers'])):
            x_in = Dense(num_dense, activation='relu')(x_in)
            if params['use_batch_norm']:
                x_in = (BatchNormalization())(x_in)
            else:
                x_in = Dropout(drop_p)(x_in)
        deep = x_in
        outs += [deep]
        
    total_out = concatenate(outs) if len(outs)>1 else outs[0]
    
    if 0<params['output_drop_out']<1:
        total_out = Dropout(params['output_drop_out'])(total_out)
    #output = Dense(params['n_output'], activation='linear')(total_out)
    output = Dense(params['n_output'], activation='sigmoid')(total_out)
    model = Model(inputs=cats+nums+[texts], output=output)
    optimizer = Adam(lr=params['lr'], decay=params['decay'])
    model.compile(loss=root_mean_squared_error, #mean_squared_error, mean_absolute_error
                  optimizer=optimizer,
                  metrics=[root_mean_squared_error])
    return model

def prepareDataFrame(df, size_cols, mean_cols):
    df['na_cnt'] = df.isnull().sum(axis=1)
    df['na_price'] = df['price'].isnull().astype('int8')
    df['item_seq_number_log1p'] = df['item_seq_number'].apply(lambda x:np.log1p(x))
    df['image_top_1'] = df['image_top_1'].fillna(df['image_top_1'].max()+1)
    
    for c in size_cols:
        print('adding size by', [c, 'activation_date'])
        df = add_group_size(df, [c, 'activation_date'])
        
    for c in mean_cols:
        print('adding mean price by', (c,))
        df = add_group_mean(df, (c,))
        
    size_cols = [c for c in df.columns if 'size_' in c]
    mean_cols = [c for c in df.columns if 'mean_' in c]
        
    for c in ['price'] + mean_cols:
        df[c] = df[c].fillna(-1)#-1
        df[c] = df[c].apply(lambda x:np.log1p(x))
    
    df = df.fillna('неизвестный')
    
    return df, size_cols, mean_cols
    
if __name__ == "__main__":
    os.listdir(DATA_DIR)

    frm = 0
    to = 1503424
    if debug:    
        frm = 0
        to = 10000
    
    testing = pd.read_csv('../input/test.csv', skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
    testdex = testing.index
    len_test = len(testing)
    
    predictors = []
    changed = False
    
    tot_filename = '/media/extend/cache/total_{}_{}'.format(frm, to)
    tot_yname = '/media/extend/cache/total_y_{}_{}'.format(frm, to)
    if os.path.exists(tot_filename) and os.path.exists(tot_yname):
        print('load from feather')
        df = pd.read_feather(tot_filename).set_index("item_id")
        y = pd.read_feather(tot_yname).set_index("item_id").deal_probability.copy()
        
        len_train = to - frm
    else:
        #tot_filename = '/media/extend/cache/total_{}_{}'.format(frm, to)
        training = pd.read_csv('../input/train.csv', skiprows=range(1,frm), nrows=to-frm, index_col = "item_id", parse_dates = ["activation_date"])
        
        
        if not debug:
            #training = training.sample(frac=0.2)
            pass
        
        traindex = training.index
        
        len_train = len(training)
        
        
        y = training.deal_probability.copy()
        training.drop("deal_probability",axis=1, inplace=True)
        y.reset_index().to_feather(tot_yname)
        
        
        print('Train shape: {} Rows, {} Columns'.format(*training.shape))
        print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
        
        df = pd.concat([training, testing], axis=0)
        del training, testing
    

    df, timechanged = calcTimeDelta(df, frm, to, predictors)
    changed |= timechanged
        
    gc.collect()
    print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
    
    print("Feature Engineering")
    df["price"] = np.log(df["price"]+0.001)
    df["price"].fillna(-999, inplace=True)
    df["image_top_1"].fillna(-999, inplace=True)
    
    
    if 'activation_date' in df.columns:
        df.loc[df['activation_date']>'2017-04-18', 'activation_date'] =  np.datetime64('2017-04-18')
        
        print("\nCreate Time Variables")
        dt = df['activation_date'].dt
        df["Weekday"] = dt.weekday.astype(np.uint8)
        df["Weekd_of_Year"] = dt.week.astype(np.uint8)
        df["DayofMonth"] = dt.day.astype(np.uint8)
        #df["DayofYear"] = dt.dayofyear.astype(np.uint16)
        df["Month"] = dt.month.astype(np.uint8)
    
        del(dt)
        gc.collect() 
    else:
        if "Weekd of Year" in df.columns:
            df.rename(columns={"Weekd of Year": "Weekd_of_Year"}, inplace = True)
    
    predictors += ["Weekday", "Weekd_of_Year", "DayofMonth", "Month", "price", "item_seq_number"]
    
    df, imgchanged = calcImgAtt(df, predictors)
    changed |= imgchanged
    
    
    # Create Validation Index and Remove Dead Variables
    #training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
    #validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
    
    
    print("\nEncode Variables")
    categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1"]
    predictors += categorical
    print("Encoding :",categorical)
    
    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in tqdm(categorical):
        df[col] = lbl.fit_transform(df[col].astype(str))
        if col == 'user_id':
            df[col] = df[col].astype(np.uint32)
        else:
            df[col] = df[col].astype(np.uint16)
        #print('max {} {}'.format(col, df[col].max()))
        
    print("\nText Features")
    
    # Feature Engineering 
    
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    count_digit = lambda s : sum(c.isdigit() for c in s)
    count_num = lambda s : sum(c.isnumeric() for c in s.split())
    
    # Meta Text Features
    textfeats = ["description", "title"]
    for cols in tqdm(textfeats):
        df[cols] = df[cols].astype(str) 
        df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        
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

    
    parse_att.checkDrop_Bulk(df, ["activation_date", "image"])
    #df.drop(["activation_date","image"],axis=1,inplace=True)
    
    feature_list = [
            (['region', 'city', 'parent_category_name'], 'count'),
            (['region', 'city', 'parent_category_name', 'category_name'], 'count'),
            
            (['region', 'city', 'parent_category_name', 'category_name', 'price'], ['count', 'mean', 'var']),
            
            (['region', 'parent_category_name', 'category_name', 'price'], ['count', 'mean','var']),
            (['parent_category_name', 'category_name', 'price'], ['count', 'mean', 'var']),
            
            (['user_id', 'price'], ['count']),
            
            (['region', 'city', 'parent_category_name', 'price'], 'count'),
            
            (['parent_category_name', 'price'], 'mean'),
            (['category_name', 'price'], 'mean'),
            (['image_top_1', 'price'], 'mean'),


            (['region', 'city', 'parent_category_name', 'category_name', 'description_num_chars'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'description_num_words'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'description_num_unique_words'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'description_words_vs_unique'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'description_punctuation'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'description_digit'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'description_num'], ['mean', 'var']),
            
            (['region', 'city', 'parent_category_name', 'category_name', 'title_num_chars'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'title_num_words'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'title_num_unique_words'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'title_words_vs_unique'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'title_punctuation'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'title_digit'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'title_num'], ['mean', 'var']),
            
            (['region', 'city', 'parent_category_name', 'category_name', 'whratio'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'area'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'laplacian'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'colorfull'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'brightness'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'median'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'rms'], ['mean', 'var']),
            (['region', 'city', 'parent_category_name', 'category_name', 'stddev'], ['mean', 'var']),
            ]
    
    
    for (selcol, how) in tqdm(feature_list):
        df, sub_changed = parse_att.calcGroupFeatureBulk(df, selcol, how, frm, to, predictors)
        changed |= sub_changed

    if not changed:
        print('df not changed')
    else:
        print('df changed, save...')
        df.reset_index().to_feather(tot_filename)
        
    str_cols = ['title', 'description']
    
    cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 
                'item_seq_number', 'user_type', 'image_top_1']
    
    df['text'] = ''
    for i, c in enumerate(df.columns):
        if c in str_cols:
            print(c)
            df['text'] += ' ' + df[c].str.lower()
            if c not in cat_cols:
                del df[c]; gc.collect()
        if c in cat_cols:
            df[c] = pd.factorize(df[c])[0]

    from keras.preprocessing import text, sequence
    print('tokenizing...')
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df['text'].values.tolist())
    
    print('getting embeddings')
    
    nb_words = min(max_features, len(tokenizer.word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for o in tqdm(open(EMB_PATH)):
        res = get_coefs(*o.rstrip().rsplit(' '), tokenizer=tokenizer)
        if res is not None:
            idx = tokenizer.word_index[res[0]]
            if idx < max_features:
                embedding_matrix[idx] = res[1]
    gc.collect()
    
    
    embedding_matrix = fill_rand_norm(embedding_matrix)

    text = df['text'].values
    del df['text']; gc.collect()
    print(df.info())
    
    text = tokenizer.texts_to_sequences(text)
    text = sequence.pad_sequences(text, maxlen=maxlen)
    
    num_feats = [x for x in predictors if x not in cat_cols]

    df_train = df[:len_train]
    text_train = text[:len_train]
    df_test = df[len_train:]
    text_test = text[len_train:]
    del text, df; gc.collect()

    X_train = get_keras_data(df_train, text_train)
    X_test = get_keras_data(df_test, text_test)
    del df_train, text_train, df_test, text_test; gc.collect()
    
    emb_cat_max = {}
    for c in cat_cols:
        emb_cat_max[c] = max(X_train[c].max(), X_test[c].max())+1
        
    
    params = {}
    params['maxlen'] = maxlen
    params['nb_words'] = nb_words
    params['embedding_matrix'] = embedding_matrix
    params['word_emb_size'] = embed_size
    params['text_emb_dropout'] = 0.2
    params['n_rnn'] = 64
    
    params['emb_cat_max'] = emb_cat_max
    params['emb_size'] = 32
    params['n_output'] = 1
    params['use_fm'] = True
    params['use_deep'] = True
    params['use_rnn'] = True
    params['use_cnn'] = True
    params['use_att'] = True
    params['use_batch_norm'] = False #replace dropout """loss: inf ???"""
    params['cnn_param'] = dict(filters=64, kernel_size=3)
    params['deep_layers'] = [512, 256, 256]
    params['drop_out'] = [0.5, 0.5, 0.5]
    params['output_drop_out'] = 0.5
    assert len(params['drop_out'])==len(params['deep_layers'])
    params['cat_feats'] = cat_cols
    #params['num_feats'] = size_cols+mean_cols+['price', 'char_len', 'word_len', 'na_price', 'na_cnt', 'item_seq_number_log1p']
    params['num_feats'] = num_feats
    params['lr'] = 0.001
    params['decay'] = 1e-7
    
    model = get_model(params)
    
    file_path = "../model/bestmixnn.hdf5"
    check_point = ModelCheckpoint(file_path, monitor='val_root_mean_squared_error', mode='min', 
                                  save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_root_mean_squared_error', patience=100, mode='min')
    
    batch_size = 2**11 * 1
    epochs = 200
    
    sample_weight = np.ones(y.shape)
    sample_weight[y<1e-7] = 1 + len(y[y<1e-7])/len(y)
    history = model.fit(X_train, y, sample_weight=sample_weight,
                        batch_size=batch_size, epochs=epochs, 
                        validation_split=0.05, verbose=1, 
                        callbacks=[check_point, early_stop])
    
    model.load_weights(file_path)
    pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    print('pred shape {}'.format(pred.shape))
    
    sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
    print('sub shape {}'.format(sub.shape))
    sub[target_col] = pred
    

    scr = min(history.history['val_root_mean_squared_error'])
    print('save to '+f'mixnn_{scr}.csv')

    str_now = datetime.now().strftime("%m-%d-%H-%M")
    kaggle_util.save_result(sub, '../result/mixnn_{}.csv'.format(str_now), 
                            competition = 'avito-demand-prediction', 
                            send = not debug, index = False)
