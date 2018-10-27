#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:51:30 2018

@author: kownse
"""

import pandas as pd 
import numpy as np 
import time 
import gc 

np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dropout, Concatenate, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import CuDNNGRU, PReLU, GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence

from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from attention_with_context import AttentionWithContext
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from Attention import Attention
from capsule import Capsule
import kaggle_util
import string

from keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing
cores = 4

from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from lightgbm_avito import calcImgAtt

from gensim.models import Word2Vec

max_seq_title_description_length = 300
max_seq_title_length = 30
max_words_title_description = 200000
EMBEDDING_DIM1 = 300
emb_size = 10
gru_size = 50

debug = 0
nfold = 5

frm = 0
to = 1503424
if debug:    
    frm = 0
    to = 1000
    
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

### rmse loss for keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) 

def clean_data(dataset):
    dataset['param_1'].fillna(value='missing', inplace=True)
    dataset['param_2'].fillna(value='missing', inplace=True)
    dataset['param_3'].fillna(value='missing', inplace=True)
    
    dataset['param_1'] = dataset['param_1'].astype(str)
    dataset['param_2'] = dataset['param_2'].astype(str)
    dataset['param_3'] = dataset['param_3'].astype(str)
    
    dataset['param123'] = (dataset['param_1']+'_'+dataset['param_2']+'_'+dataset['param_3']).astype(str)
    del dataset['param_2'], dataset['param_3']
    gc.collect()
    
    return dataset

def preprocess_dataset(dataset):
    
    t1 = time.time()
    print("Filling Missing Values.....")
    
    dataset['price'] = dataset['price'].fillna(0).astype('float32')
    
    print("Casting data types to type Category.......")
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['parent_category_name'] = dataset['parent_category_name'].astype('category')
    dataset['region'] = dataset['region'].astype('category')
    dataset['city'] = dataset['city'].astype('category')
    
    dataset = clean_data(dataset)
        
    print("PreProcessing Function completed.")
    
    return dataset

def get_numcols(dataset):
    non_num_cols = ['seq_description', 'seq_title', 'user_id'] + emb_cols
    num_cols = []
    
    X = {}
    for c in dataset.columns:         
        if c not in non_num_cols:
            num_cols.append(c)
            
    return num_cols


def num_log(df):
    df['price'] = np.log1p(df['price'])
    df['avg_days_up_user'] = np.log1p(df['avg_days_up_user'])
    df['avg_times_up_user'] = np.log1p(df['avg_times_up_user'])
    df['n_user_items'] = np.log1p(df['n_user_items'])
    df['item_seq_number'] = np.log(df['item_seq_number'])

    num_cols = get_numcols(df)
    #print(num_cols)
    #print(df[num_cols].head())
    #print(df[num_cols].isnull().sum())
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

def keras_fit(train):
    
    t1 = time.time()
    train['title_description']= (train['title']+" "+train['description']).astype(str)
    
    print("Start Tokenization.....")
    tokenizer = kaggle_util.get_text_tokenizer(train, 'title_description', max_words_title_description)
    
    regional = pd.read_csv('../input/regional.csv', index_col=0)
    regional.index = regional.index.str.lower()

    train['region'] = train['region'].apply(lambda x : region_map[x])
    train['region'] = train['region'].str.lower()
    train["reg_dense"] = train['region'].apply(lambda x: regional.loc[x,"Density_of_region(km2)"])
    train["rural"] = train['region'].apply(lambda x: regional.loc[x,"Rural_%"])
    train["reg_Time_zone"] = train['region'].apply(lambda x: regional.loc[x,"Time_zone"])
    train["reg_Population"] = train['region'].apply(lambda x: regional.loc[x,"Total_population"])
    train["reg_Urban"] = train['region'].apply(lambda x: regional.loc[x,"Urban%"])

    dict_encoder = {}
    for col in emb_cols:
        encoder = LabelEncoder()
        encoder.fit(train[col])
        dict_encoder[col] = encoder
    
    print("Fit on Train Function completed.")
    
    return train, tokenizer, dict_encoder

def deal_text_feature(dataset):
    t1 = time.time()
    
    #dataset['title_description'].fillna('', inplace=True)
    #dataset['seq_title_description']= tokenizer.texts_to_sequences(dataset.title_description.str.lower())
    
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    count_digit = lambda s : sum(c.isdigit() for c in s)
    count_num = lambda s : sum(c.isnumeric() for c in s.split())
    
    dataset['description'].fillna('unknown', inplace=True)
    dataset['title'].fillna('unknown', inplace=True)
    
    predictors = []
    textfeats = ["description", "title"]
    for cols in tqdm(textfeats):
        dataset[cols] = dataset[cols].str.lower()
           
        att_name = cols + '_num_words'
        predictors.append(att_name)
        dataset[att_name] = dataset[cols].apply(lambda comment: len(comment.split())).astype(np.uint16) # Count number of Words
            
        att_name = cols + '_num_unique_words'
        predictors.append(att_name)
        dataset[att_name] = dataset[cols].apply(lambda comment: len(set(w for w in comment.split()))).astype(np.uint16)
            
        att_name = cols + '_words_vs_unique'
        predictors.append(att_name)
        dataset[att_name] = (dataset[cols+'_num_unique_words'] / dataset[cols+'_num_words']).astype(np.float32) # Count Unique Words
            
        att_name = cols + '_punctuation'
        predictors.append(att_name)
        dataset[att_name] = dataset[cols].apply(count, args=(string.punctuation,)).astype(np.uint16)
           
        att_name = cols + '_num'
        predictors.append(att_name)
        dataset[att_name] = dataset[cols].apply(count_num).astype(np.uint16)
            
    dataset['title_desc_len_ratio'] = dataset['title_num_words']/dataset['description_num_words']
    #dataset['desc_num_ratio'] = dataset['description_num']/dataset['description_num_words']
    predictors += ['title_desc_len_ratio']#, 'desc_num_ratio']
    
    dataset['seq_description']= tokenizer.texts_to_sequences(dataset.description.str.lower())
    dataset['seq_title']= tokenizer.texts_to_sequences(dataset.title.str.lower())
    
    print("Transform done for test")
    print("Time taken for Sequence Tokens is"+str(time.time()-t1))
    
    del dataset['title_description']
    del dataset['description'], dataset['title']
    gc.collect()
    
    return dataset, predictors

def keras_train_transform(dataset):
    print('transform...')
    dataset, txt_stats = deal_text_feature(dataset)
    
    for key in dict_encoder.keys():
        #print(key)
        dataset[key] = dict_encoder[key].transform(dataset[key])
    
    dataset = kaggle_util.reduce_mem_usage(dataset)
    print("Transform on test function completed.")
    
    dataset = num_log(dataset)
    
    return dataset, txt_stats
    
def get_keras_data(dataset):
    X = {}
    for c in dataset.columns:
        if c in ['item_id', 'user_id']:
            continue   
        elif c == 'seq_description':
            X[c] = pad_sequences(dataset[c], maxlen=max_seq_title_description_length)
        elif c == 'seq_title':
            X[c] = pad_sequences(dataset[c], maxlen=max_seq_title_length)
        
        #if c == 'seq_title_description':
        #    X[c] = pad_sequences(dataset[c], maxlen=max_seq_title_description_length)
        else:
            X[c] = dataset[c].values
            

    print("Data ready for Vectorization")
    
    return X

def RNN_model(emb_cols, num_cols):
#def RNN_model():

    #Inputs
    #seq_title_description = Input(shape=[max_seq_title_description_length], name="seq_title_description")
    seq_description = Input(shape=[max_seq_title_description_length], name="seq_description")
    seq_title = Input(shape=[max_seq_title_length], name="seq_title")
    
    emb_inputs = []
    emb_layers = []
    for col in emb_cols:
        emb_input = Input(shape=[1], name=col)
        emb_inputs.append(emb_input)
        
        emb_layer = Embedding(dict_emb_max[col], emb_size)(emb_input)
        emb_layers.append(Flatten()(emb_layer))
        
    num_inputs = []
    for col in num_cols:
        num_inputs.append(Input(shape=[1], name=col))
    
    #Embeddings layers
    
    
    #emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title_description)
    emb_seq_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_description)
    emb_seq_title = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title)
    
    """
    rnn_layer1 = Bidirectional(CuDNNGRU(gru_size, return_sequences=True))(emb_seq_title_description)
    rnn_layer1 = AttentionWithContext()(rnn_layer1)
    
    Routings = 6
    Num_capsule = 10
    Dim_capsule = 16
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                          share_weights=True)(rnn_layer1)
    rnn_layer1 = Flatten()(capsule)
    """
    
    #rnn_layer1 = CuDNNGRU(gru_size)(emb_seq_title_description)
    rnn_layer1 = CuDNNGRU(gru_size)(emb_seq_description)
    rnn_layer2 = CuDNNGRU(gru_size)(emb_seq_title)
    #rnn_layer1 = CuDNNGRU(gru_size, return_sequences=True)(emb_seq_description)
    #rnn_layer2 = CuDNNGRU(gru_size, return_sequences=True)(emb_seq_title)
    #rnn_layer1 = Attention()(rnn_layer1)
    #rnn_layer2 = Attention()(rnn_layer2)
    
    
    """
    rnn_layer1 = Bidirectional(CuDNNGRU(gru_size, return_sequences=True))(emb_seq_title_description)
    avg_tensor = GlobalAveragePooling1D()(rnn_layer1)
    max_tensor = GlobalMaxPooling1D()(rnn_layer1)
    rnn_layer1 = Concatenate()([avg_tensor, max_tensor])
    """
    #main layer
    main_l = concatenate([rnn_layer1, rnn_layer2] + emb_layers + num_inputs)
        
    #main_l = Dropout(0.1)(Dense(512,activation='relu') (main_l))
    #main_l = Dropout(0.1)(Dense(64,activation='relu') (main_l))
    
    
    main_l = BatchNormalization()(main_l)
    #main_l = Dropout(0.1)(main_l) # maybe bad 
    main_l = Dense(512)(main_l)
    main_l = PReLU()(main_l)
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.1)(main_l)
    main_l = Dense(256)(main_l)
    main_l = PReLU()(main_l)
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.1)(main_l)
    
    
    #output
    output = Dense(3067,activation="softmax") (main_l)
    
    #model
    inputs = [seq_description, seq_title] + emb_inputs + num_inputs
    model = Model(inputs, output)
                  
    model.compile(optimizer = 'adam',
                  loss= sparse_categorical_crossentropy,
                  metrics = [sparse_categorical_crossentropy])
    return model

def rmse(y, y_pred):

    Rsum = np.sum((y - y_pred)**2)
    n = y.shape[0]
    RMSE = np.sqrt(Rsum/n)
    return RMSE 

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

if __name__ == "__main__":
    
    dtypes_train = {
                'price': 'float32',
                'item_seq_number': 'uint32'
    }

    # No user_id
    train_cols = ['item_id', 'user_id', 'region', 'city', 'parent_category_name', 'category_name',
       'param_1', 'param_2', 'param_3', 'title', 'description', 'price',
       'item_seq_number', 'user_type', 'image_top_1']
    train = pd.read_csv("../input/train.csv", skiprows=range(1,frm), nrows=to-frm, 
                        dtype = dtypes_train, 
                        index_col = "item_id",
                       usecols = train_cols)

    test = pd.read_csv("../input/test.csv", skiprows=range(1,frm), nrows=to-frm, dtype = dtypes_train, index_col = "item_id", usecols = train_cols)
    testdex = test.index

    df = pd.concat([train, test], axis = 0)
    testdex = df[pd.isnull(df['image_top_1'])].index
    traindex = df[pd.notnull(df['image_top_1'])].index

    train = df.loc[traindex]
    test = df.loc[testdex]

    y_train = np.array(train['image_top_1'])

    len_train = len(train)
    train = pd.concat([train,test])
    del train['image_top_1']
    gc.collect()

    train_features = pd.read_csv('../input/aggregated_features.csv')
    train = train.merge(train_features, on = ['user_id'], how = 'left')
    del train_features
    gc.collect()

    train['avg_days_up_user'] = train['avg_days_up_user'].fillna(0).astype('uint32')
    train['avg_times_up_user'] = train['avg_times_up_user'].fillna(0).astype('uint32')
    train['n_user_items'] = train['n_user_items'].fillna(0).astype('uint32')

    emb_cols = ['region', 'city', 'category_name', 'parent_category_name', 'user_type',
                'param_1', 'param123', 'reg_Time_zone']

    train = preprocess_dataset(train)
    train, tokenizer, dict_encoder = keras_fit(train)
    train, txt_stats = keras_train_transform(train)
    print("Tokenization done and TRAIN READY FOR Validation splitting")

    # Calculation of max values for Categorical fields 

    dict_emb_max = {}
    for col in emb_cols:
        dict_emb_max[col] = train[col].max() + 2

    #del train['item_id'], 
    del train['user_id']
    gc.collect()

    EMBEDDING_FILE1 = '../input/wiki.ru.vec'
    embedding_matrix1, vocab_size = kaggle_util.build_emb_matrix_from_tokenizer(tokenizer, EMBEDDING_FILE1, EMBEDDING_DIM1)

    test = train[len_train:]
    train = train[:len_train]
    X_test = get_keras_data(test)
    num_cols = get_numcols(test)
    print('test shape: {}'.format(test.shape))
    print('num_cols:')
    print(num_cols)

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    import time 

    skf = KFold(n_splits = nfold)
    Kfold_preds_final = []
    k = 0
    RMSE = []
    
    preds_total = None

    for train_idx, test_idx in skf.split(train.values, y_train):

        print("Number of Folds.."+str(k+1))
        #K Fold Split 

        X_train1, X_test1 = train.iloc[train_idx], train.iloc[test_idx]
        print('input shape: ', X_train1.shape, X_test1.shape)
        y_train1, y_test1 = y_train[train_idx], y_train[test_idx]
        print(y_train1.shape, y_test1.shape)
        gc.collect()

        X_train_final = get_keras_data(X_train1)
        X_test_final = get_keras_data(X_test1)


        # Initialize a new Model for Current FOLD 
        epochs = 1
        batch_size = 512 * 3
        steps = (int(train.shape[0]/batch_size))*epochs
        lr_init, lr_fin = 0.009, 0.0045
        lr_decay = 0.0001#exp_decay(lr_init, lr_fin, steps)
        #modelRNN = RNN_model()
        modelRNN = RNN_model(emb_cols, num_cols)
        K.set_value(modelRNN.optimizer.lr, lr_init)
        K.set_value(modelRNN.optimizer.decay, lr_decay)

        # Fit the NN Model 
        file_path = "../model/bestrnnimg_{}.hdf5".format(k)
        check_point = ModelCheckpoint(file_path, monitor='val_sparse_categorical_crossentropy', mode='min', 
                                      save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_sparse_categorical_crossentropy', patience=4, mode='min')
        hist = modelRNN.fit(X_train_final, y_train1, batch_size=batch_size, epochs=100, 
                            validation_data=(X_test_final, y_test1), verbose=1,
                            callbacks=[check_point, early_stop])

        del X_train_final
        gc.collect()
        """
        del modelRNN
        gc.collect()
        K.clear_session()

        modelRNN = RNN_model(emb_cols, num_cols)
        K.set_value(modelRNN.optimizer.lr, lr_init)
        K.set_value(modelRNN.optimizer.decay, lr_decay)
        """
        modelRNN.load_weights(file_path)
        del X_test_final
        del y_train1, y_test1
        gc.collect()

        preds = modelRNN.predict(X_test, batch_size = batch_size, verbose = 1)
        del modelRNN
        gc.collect()
        
        if preds_total is None:
            preds_total = preds
        else:
            preds_total += preds
        
        print("Number of folds completed...."+str(k))
        #print(Kfold_preds_final[k][0:10])
        k += 1
        K.clear_session()
     
    preds_total /= k
    actual_cnt = 0
    classes = np.zeros(shape=np.argmax(preds,axis = 1).shape)
    for i in range(preds.shape[0]):
        if np.max(preds[i]) > 0.1:
            actual_cnt += 1
            classes[i] = np.argmax(preds[i])
        else:
            classes[i] = np.nan

    result = pd.DataFrame({'image_top_1':classes}, index=testdex)
    result.to_csv('../input/advance_imgtop.csv')
    print('predicted {} / {}'.format(actual_cnt, preds_total.shape[0]))