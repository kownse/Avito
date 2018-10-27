#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:10:52 2018

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
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model

from keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

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

import kaggle_util
from lightgbm_avito import calcImgAtt
from tqdm import tqdm

debug = 0


max_seq_title_description_length = 200
max_words_title_description = 200000
if debug:
    max_words_title_description = 2000

### rmse loss for keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) 
    
def get_keras_data(df):
    X = {}
    for c in df.columns:
        if c != 'seq_title_description':
            X[c] = df[c].values
        else:
            X[c] = pad_sequences(df[c], maxlen=max_seq_title_description_length)
    return X

def prepare_data(frm, to):
    # No user_id
    use_cols = ['item_id', 'user_id', 'image_top_1', 'region', 'city', 'user_type', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price', 'item_seq_number', 'deal_probability', 'image', 'activation_date']
    train = pd.read_csv("../input/train.csv", skiprows=range(1,frm), nrows=to-frm, parse_dates=["activation_date"], usecols = use_cols, dtype = dtypes_train)
    use_cols_test = ['item_id', 'user_id', 'image_top_1', 'region', 'city', 'user_type', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price', 'item_seq_number', 'image', 'activation_date']
    test = pd.read_csv("../input/test.csv", skiprows=range(1,frm), nrows=to-frm, parse_dates=["activation_date"], usecols = use_cols_test)
    ntrain = train.shape[0]
    y_train = np.array(train['deal_probability'])
    
    df = pd.concat([train, test], axis = 0)
    
    del test, train
    gc.collect()
    
    df["weekday"] = df['activation_date'].dt.weekday.astype(np.uint8)
    df.drop('activation_date', axis=1, inplace=True)
    
    train_features = pd.read_csv('../input/aggregated_features.csv')
    df = df.merge(train_features, on = ['user_id'], how = 'left')
    agg_att = list(train_features.columns)
    agg_att.remove('user_id')
    del train_features
    gc.collect()
    
    
    img_att = []
    df, imgchanged = calcImgAtt(df, img_att)
    df.drop('image', axis=1, inplace=True)
    
    for col in (agg_att + img_att + ['price']):
        print(col)
        df[col] = np.log1p(df[col]).fillna(0)
        df[col] = df[col].astype(np.float32)
        
        
    df = df.rename({'image_top_1':'image_code'}, axis='columns')
    categorical = ["user_id", "user_type", "region","city","parent_category_name",
                   "category_name","user_type","image_code",
                   "param_1","param_2","param_3"]
    # Encoder:
    for col in tqdm(categorical):
        df[col].fillna('Unknown')
        lbl = LabelEncoder()
        df[col] = lbl.fit_transform(df[col].astype(str))
        if col == 'user_id':
            df[col] = df[col].astype(np.uint32)
        else:
            df[col] = df[col].astype(np.uint16)
            
    df['title_description']= (df['title']+" "+df['description']).astype(str)
    del df['description'], df['title']
    gc.collect()
    
    print("Start Tokenization.....")
    
        
    tokenizer = text.Tokenizer(num_words = max_words_title_description)
    all_text = np.hstack([df['title_description'].str.lower()])
    tokenizer.fit_on_texts(all_text)
    del all_text
    gc.collect()
    
    t1 = time.time()
    df['seq_title_description']= tokenizer.texts_to_sequences(df.title_description.str.lower())
    print("Transform done for test")
    print("Time taken for Sequence Tokens is "+str(time.time()-t1))
    del df['title_description']
    gc.collect()
    
    
    
    del df['deal_probability']
    gc.collect()
    
    print("Tokenization done and TRAIN READY FOR Validation splitting")
    
    return df, y_train, ntrain, tokenizer


# Loading Train data - No Params, No Image data 
dtypes_train = {
    'price': 'float32',
    'deal probability': 'float32',
    'item_seq_number': 'uint32'
}



frm = 0
to = 1503424
if debug:    
    frm = 0
    to = 10000


df, y_train, ntrain, tokenizer = prepare_data(frm, to)

# Calculation of max values for Categorical fields 

max_userid = df.user_id.max() + 1
max_region = df.region.max() + 1
max_city= df.city.max() + 1
max_parent_category_name = df.parent_category_name.max() + 1
max_category_name = df.category_name.max() + 1
max_usertype = df.user_type.max() + 1
max_image_code = df.image_code.max() + 1

max_param_1 = df.param_1.max() + 1
max_param_2 = df.param_1.max() + 1
max_param_3 = df.param_1.max() + 1
#max_week = np.max(train.week.max())+2
#max_day = np.max(train.day.max())+2
#max_wday = np.max(train.wday.max())+2
max_weekday = np.max(df.weekday.max())+2


EMBEDDING_DIM1 = 300
EMBEDDING_FILE1 = '../input/wiki.ru.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

vocab_size = len(tokenizer.word_index)+2
EMBEDDING_DIM1 = 300# this is from the pretrained vectors
embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
print(embedding_matrix1.shape)
# Creating Embedding matrix 
c = 0 
c1 = 0 
w_Y = []
w_No = []
for word, i in tokenizer.word_index.items():
    if word in embeddings_index1:
        c +=1
        embedding_vector = embeddings_index1[word]
        w_Y.append(word)
    else:
        embedding_vector = None
        w_No.append(word)
        c1 +=1
    if embedding_vector is not None:    
        embedding_matrix1[i] = embedding_vector

print(c,c1, len(w_No), len(w_Y))
print(embedding_matrix1.shape)
del embeddings_index1
gc.collect()

print(" FAST TEXT DONE")

def RNN_model():

    #Inputs
    inputs = []
    category_name = Input(shape=[1], name="category_name")
    city = Input(shape=[1], name="city")
    image_code = Input(shape=[1], name="image_code")
    item_seq_number = Input(shape = [1], name = 'item_seq_number')
    
    inputs += [category_name, city, image_code, item_seq_number]
    
    param_1 = Input(shape=[1], name="param_1")
    param_2 = Input(shape=[1], name="param_2")
    param_3 = Input(shape=[1], name="param_3")
    
    inputs += [param_1, param_2, param_3]
  
    parent_category_name = Input(shape=[1], name="parent_category_name")
    price = Input(shape=[1], name="price")
    region = Input(shape=[1], name="region")
    user_id = Input(shape=[1], name="user_id")
    user_type = Input(shape=[1], name="user_type")
    weekday = Input(shape=[1], name="weekday")
    
    inputs += [parent_category_name, price, region, user_id, user_type, weekday]

    avg_days_up_user = Input(shape=[1], name="avg_days_up_user")
    avg_times_up_user = Input(shape=[1], name="avg_times_up_user")
    n_user_items = Input(shape=[1], name="n_user_items")
    
    inputs += [avg_days_up_user, avg_times_up_user, n_user_items]
    
    whratio = Input(shape=[1], name="whratio")
    area = Input(shape=[1], name="area")
    laplacian = Input(shape=[1], name="laplacian")
    colorfull = Input(shape=[1], name="colorfull")
    brightness = Input(shape=[1], name="brightness")
    median = Input(shape=[1], name="median")
    rms = Input(shape=[1], name="rms")
    stddev = Input(shape=[1], name="stddev")
    
    inputs += [whratio, area, laplacian, colorfull, brightness, median, rms, stddev]
    
    seq_title_description = Input(shape=[max_seq_title_description_length], name="seq_title_description")
    inputs += [seq_title_description]
    
    ["user_id", "user_type","region","city","parent_category_name",
               "category_name","image_code",
               "param_1","param_2","param_3"]
    #Embeddings layers
    
    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title_description)
    rnn_layer1 = GRU(50) (emb_seq_title_description)
    
    """
    emb_size = 32
    emb_user_id = Embedding(max_userid, emb_size)(user_id)
    emb_user_type = Embedding(max_usertype, emb_size)(user_type)
    emb_region = Embedding(max_region, emb_size)(region)
    emb_city = Embedding(max_city, emb_size)(city)
    emb_parent_category_name = Embedding(max_parent_category_name, emb_size)(parent_category_name)
    emb_category_name = Embedding(max_category_name, emb_size)(category_name)
    emb_image_code = Embedding(max_image_code, emb_size)(image_code)
    emb_param_1 = Embedding(max_param_1, emb_size)(param_1)
    emb_param_2 = Embedding(max_param_2, emb_size)(param_2)
    emb_param_3 = Embedding(max_param_3, emb_size)(param_3)
    """
    emb_size = 10
    emb_user_id = Embedding(vocab_size, emb_size)(user_id)
    emb_user_type = Embedding(vocab_size, emb_size)(user_type)
    emb_region = Embedding(vocab_size, emb_size)(region)
    emb_city = Embedding(vocab_size, emb_size)(city)
    emb_parent_category_name = Embedding(vocab_size, emb_size)(parent_category_name)
    emb_category_name = Embedding(vocab_size, emb_size)(category_name)
    emb_image_code = Embedding(vocab_size, emb_size)(image_code)
    emb_param_1 = Embedding(vocab_size, emb_size)(param_1)
    emb_param_2 = Embedding(vocab_size, emb_size)(param_2)
    emb_param_3 = Embedding(vocab_size, emb_size)(param_3)
    
    
    #main layer
    main_l = concatenate([
          rnn_layer1
        , Flatten() (emb_user_id)
        , Flatten() (emb_user_type)
        , Flatten() (emb_region)
        , Flatten() (emb_city)
        , Flatten() (emb_parent_category_name)
        , Flatten() (emb_category_name)
        , Flatten() (emb_image_code)
        , Flatten() (emb_param_1)
        , Flatten() (emb_param_2)
        , Flatten() (emb_param_3)
        , weekday
        , avg_days_up_user
        , avg_times_up_user
        , n_user_items
        , price
        , item_seq_number
        , whratio
        , area
        , laplacian
        , colorfull
        , brightness
        , median
        , rms
        , stddev
    ])
    
    main_l = Dropout(0.5)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(0.2)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="sigmoid") (main_l)
    
    #model
    model = Model(inputs, output)
    model.compile(optimizer = 'adam',
                  loss= root_mean_squared_error,
                  metrics = [root_mean_squared_error])
    return model

def rmse(y, y_pred):

    Rsum = np.sum((y - y_pred)**2)
    n = y.shape[0]
    RMSE = np.sqrt(Rsum/n)
    return RMSE 

def eval_model(model, X_test1):
    val_preds = model.predict(X_test1)
    y_pred = val_preds[:, 0]
    
    y_true = np.array(y_test1)
    
    yt = pd.DataFrame(y_true)
    yp = pd.DataFrame(y_pred)
    
    print(yt.isnull().any())
    print(yp.isnull().any())
    
    v_rmse = rmse(y_true, y_pred)
    print(" RMSE for VALIDATION SET: "+str(v_rmse))
    return v_rmse

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

def predictions(model):
    import time
    t1 = time.time()
    def load_test():
        for df in pd.read_csv('../input/test.csv', chunksize= 250000):
            yield df

    item_ids = np.array([], dtype=np.int32)
    preds= np.array([], dtype=np.float32)

    i = 0 
    
    for df in load_test():
    
        i +=1
        print(df.dtypes)
        item_id = df['item_id']
        print(" Chunk number is "+str(i))
    
        test = preprocess_dataset(df)
    
        train_features = pd.read_csv('../input/aggregated_features.csv')
        test = test.merge(train_features, on = ['user_id'], how = 'left')
        del train_features
        gc.collect()
    
        print(test.dtypes)
        
        test['avg_days_up_user'] = test['avg_days_up_user'].fillna(0).astype('uint32')
        test['avg_times_up_user'] = test['avg_times_up_user'].fillna(0).astype('uint32')
        test['n_user_items'] = test['n_user_items'].fillna(0).astype('uint32')
        test = keras_test_transform(test)
        del df
        gc.collect()
    
        print(test.dtypes)
    
        X_test = get_keras_data(test)
        del test 
        gc.collect()
    
        Batch_Size = 512*3
        preds1 = modelRNN.predict(X_test, batch_size = Batch_Size, verbose = 1)
        print(preds1.shape)
        del X_test
        gc.collect()
        print("RNN Prediction is done")

        preds1 = preds1.reshape(-1,1)
        #print(predsl.shape)
        preds1 = np.clip(preds1, 0, 1)
        print(preds1.shape)
        item_ids = np.append(item_ids, item_id)
        print(item_ids.shape)
        preds = np.append(preds, preds1)
        print(preds.shape)
        
    print("All chunks done")
    t2 = time.time()
    print("Total time for Parallel Batch Prediction is "+str(t2-t1))
    return preds 


#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time 

if debug:
    splits = 3
else:
    splits = 5
skf = KFold(n_splits = splits)
Kfold_preds_final = []
k = 0
RMSE = []

train = df[:ntrain]
test = df[ntrain:]

train.drop('item_id', axis=1, inplace=True)

X_test = get_keras_data(test)

cnt = 0
for train_idx, test_idx in skf.split(train, y_train):
    cnt += 1
    print("Number of Folds.."+str(k+1))
    
    # Initialize a new Model for Current FOLD 
    if debug:
        epochs = 1
    else:
        epochs = 3
    batch_size = 512 * 3
    steps = (int(train.shape[0]/batch_size))*epochs
    lr_init, lr_fin = 0.009, 0.0045
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    modelRNN = RNN_model()
    K.set_value(modelRNN.optimizer.lr, lr_init)
    K.set_value(modelRNN.optimizer.decay, lr_decay)

    #K Fold Split 
    
    X_train1, X_test1 = train.iloc[train_idx], train.iloc[test_idx]
    print(X_train1.shape, X_test1.shape)
    y_train1, y_test1 = y_train[train_idx], y_train[test_idx]
    print(y_train1.shape, y_test1.shape)
    gc.collect()
    
    print(type(X_train1))
    print(X_train1.shape)
    
    X_train_f = get_keras_data(X_train1)
    X_test_f = get_keras_data(X_test1)
    

    # Fit the NN Model 
    file_path = "../model/bestrnn_{}.hdf5".format(cnt)
    """
    for i in range(3):
        hist = modelRNN.fit(X_train_f, y_train1, batch_size=batch_size+(batch_size*(2*i)), epochs=epochs, validation_data=(X_test_f, y_test1), verbose=1)
    """
    """
    
    check_point = ModelCheckpoint(file_path, monitor='val_root_mean_squared_error', mode='min', 
                                  save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_root_mean_squared_error', patience=3, mode='min')
    hist = modelRNN.fit(X_train_f, y_train1, batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_test_f, y_test1), verbose=1,
                        callbacks=[check_point, early_stop])
    """
    check_point = ModelCheckpoint(file_path, monitor='val_root_mean_squared_error', mode='min', 
                                  save_best_only=False, verbose=1)
    hist = modelRNN.fit(X_train_f, y_train1, batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_test_f, y_test1), verbose=1,
                        callbacks=[check_point])

    del X_train_f
    gc.collect()
    
    # Predict test set for Kth Fold 
    #preds = predictions(modelRNN)
    #modelRNN.load_weights(file_path)
    
    # Print RMSE for Validation set for Kth Fold 
    v_rmse = eval_model(modelRNN, X_test_f)
    RMSE.append(v_rmse)
    
    del X_test_f
    del y_train1, y_test1
    gc.collect()
    
    preds = modelRNN.predict(X_test, batch_size = batch_size, verbose = 1)
    
    del modelRNN 
    gc.collect()

    print("Predictions done for Fold "+str(k))
    print(preds.shape)
    Kfold_preds_final.append(preds)
    del preds
    gc.collect()
    print("Number of folds completed...."+str(len(Kfold_preds_final)))
    print(Kfold_preds_final[k][0:10])
    
    from keras import backend as K
    K.clear_session()

print("All Folds completed"+str(k+1))   
print("RNN FOLD MODEL Done")

pred_final1 = np.average(Kfold_preds_final, axis =0) # Average of all K Folds
print(pred_final1.shape)

min_value = min(RMSE)
RMSE_idx = RMSE.index(min_value)
print(RMSE_idx)
pred_final2 = Kfold_preds_final[RMSE_idx]
print(pred_final2.shape)

#del Kfold_preds_final, train1
gc.collect()

test_cols = ['item_id']
test = pd.read_csv('../input/test.csv', skiprows=range(1,frm), nrows=to-frm, usecols = test_cols)

# using Average of KFOLD preds 

submission1 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission1['item_id'] = test['item_id']
submission1['deal_probability'] = pred_final1

print("Check Submission NOW!!!!!!!!@")
#submission1.to_csv("Avito_Shanth_RNN_AVERAGE.csv", index=False)
kaggle_util.save_result(submission1, '../result/rnn_avg.csv', competition = 'avito-demand-prediction', send = not debug, index = False)

# Using KFOLD preds with Minimum value 
submission2 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission2['item_id'] = test['item_id']
submission2['deal_probability'] = pred_final2

print("Check Submission NOW!!!!!!!!@")
#submission2.to_csv("Avito_Shanth_RNN_MIN.csv", index=False)
kaggle_util.save_result(submission2, '../result/rnn_min.csv', competition = 'avito-demand-prediction', send = False, index = False)