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
from capsule import Capsule

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
from lightgbm_avito import calcImgAtt, shrink

from gensim.models import Word2Vec

max_seq_title_description_length = 300
max_seq_title_length = 25
max_words_title_description = 200000
EMBEDDING_DIM1 = 300
emb_size = 10
gru_size = 50

debug = 0

frm = 0
to = 1503424
if debug:    
    frm = 0
    to = 10000

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
    
    dataset['image_top_1'] = dataset['image_top_1'].fillna('missing')
    dataset['image_code'] = dataset['image_top_1'].astype('str')
    del dataset['image_top_1']
    gc.collect()

    #dataset['week'] = pd.to_datetime(dataset['activation_date']).dt.week.astype('uint8')
    #dataset['day'] = pd.to_datetime(dataset['activation_date']).dt.day.astype('uint8')
    #dataset['wday'] = pd.to_datetime(dataset['activation_date']).dt.dayofweek.astype('uint8')
    del dataset['activation_date']
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

def num_log(df):
    df['price'] = np.log1p(df['price'])
    df['avg_days_up_user'] = np.log1p(df['avg_days_up_user'])
    df['avg_times_up_user'] = np.log1p(df['avg_times_up_user'])
    df['n_user_items'] = np.log1p(df['n_user_items'])
    df['item_seq_number'] = np.log(df['item_seq_number'])

    df['whratio'] = np.log1p(df['whratio'])
    df['laplacian'] = np.log1p(df['laplacian'])
    df['colorfull'] = np.log1p(df['colorfull'])
    df['brightness'] = np.log1p(df['brightness'])
    df['median'] = np.log1p(df['median'])
    df['rms'] = np.log1p(df['rms'])
    df['stddev'] = np.log1p(df['stddev'])

    cols = ['price', 'avg_days_up_user', 'avg_times_up_user', 'n_user_items', 'item_seq_number',
            'whratio', 'laplacian', 'colorfull', 'brightness', 'median','rms', 'stddev',
            ]
    
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    
    return df

def keras_fit(train):
    
    t1 = time.time()
    train['title_description']= (train['title']+" "+train['description']).astype(str)
    del train['description'], train['title']
    gc.collect()
    
    print("Start Tokenization.....")
    tokenizer = text.Tokenizer(num_words = max_words_title_description)
    all_text = np.hstack([train['title_description'].str.lower()])
    tokenizer.fit_on_texts(all_text)
    del all_text
    gc.collect()
    
    print("Loading Test for Label Encoding on Train + Test")
    use_cols_test = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'image_top_1', 'activation_date', 'image']
    #test = pd.read_csv("../input/test.csv", skiprows=range(1,frm), nrows=to-frm, usecols = use_cols_test)
    test = pd.read_csv("../input/test.csv", usecols = use_cols_test)
    
    test = clean_data(test)
    
    ntrain = train.shape[0]
    DF = pd.concat([train, test], axis = 0)
    del train, test
    gc.collect()
    print(DF.shape)
    
    dict_encoder = {}
    for col in emb_cols:
        encoder = LabelEncoder()
        encoder.fit(DF[col])
        dict_encoder[col] = encoder
    
    train = DF[0:ntrain]
    del DF 
    gc.collect()
    
    train = num_log(train)
    print("Fit on Train Function completed.")
    
    return train, tokenizer, dict_encoder

def deal_text_feature(dataset):
    t1 = time.time()
    dataset['title_description'].fillna('', inplace=True)
    dataset['seq_title_description']= tokenizer.texts_to_sequences(dataset.title_description.str.lower())
    """
    dataset['description'].fillna('', inplace=True)
    dataset['title'].fillna('', inplace=True)
    dataset['seq_description']= tokenizer.texts_to_sequences(dataset.description.str.lower())
    dataset['seq_title']= tokenizer.texts_to_sequences(dataset.title.str.lower())
    """
    print("Transform done for test")
    print("Time taken for Sequence Tokens is"+str(time.time()-t1))
    
    del dataset['title_description']
    #del dataset['description'], dataset['title']
    gc.collect()
    
    return dataset

def keras_train_transform(dataset):
    
    dataset = deal_text_feature(dataset)
    
    for key in dict_encoder.keys():
        
        dataset[key] = dict_encoder[key].transform(dataset[key])
    
    dataset = shrink(dataset)
    print("Transform on test function completed.")
    
    return dataset
    
def keras_test_transform(dataset):
    
    dataset['title_description']= (dataset['title']+" "+dataset['description']).astype(str)
    del dataset['description'], dataset['title']
    gc.collect()
    
    dataset = deal_text_feature(dataset)

    for key in dict_encoder.keys():
        print(key)
        dataset[key] = dict_encoder[key].transform(dataset[key])
    
    dataset = num_log(dataset)
    print("Transform on test function completed.")
    
    dataset = shrink(dataset)
    
    return dataset
    
def get_keras_data(dataset):

    X = {}
    for c in dataset.columns:
        """
        if c == 'seq_description':
            X[c] = pad_sequences(dataset[c], maxlen=max_seq_title_description_length)
        elif c == 'seq_title':
            X[c] = pad_sequences(dataset[c], maxlen=max_seq_title_length)
        """
        if c == 'seq_title_description':
            X[c] = pad_sequences(dataset[c], maxlen=max_seq_title_description_length)
        else:
            X[c] = dataset[c].values

    print("Data ready for Vectorization")
    
    return X

def RNN_model():

    #Inputs
    seq_title_description = Input(shape=[max_seq_title_description_length], name="seq_title_description")
    #seq_description = Input(shape=[max_seq_title_description_length], name="seq_description")
    #seq_title = Input(shape=[max_seq_title_length], name="seq_title")
    
    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")
    param_1 = Input(shape=[1], name="param_1")
    param123 = Input(shape=[1], name="param123")
    image_code = Input(shape=[1], name="image_code")
    price = Input(shape=[1], name="price")
    item_seq_number = Input(shape = [1], name = 'item_seq_number')
    avg_days_up_user = Input(shape=[1], name="avg_days_up_user")
    avg_times_up_user = Input(shape=[1], name="avg_times_up_user")
    n_user_items = Input(shape=[1], name="n_user_items")
    
    
    whratio = Input(shape=[1], name="whratio")
    laplacian = Input(shape=[1], name="laplacian")
    colorfull = Input(shape=[1], name="colorfull")
    brightness = Input(shape=[1], name="brightness")
    median = Input(shape=[1], name="median")
    rms = Input(shape=[1], name="rms")
    stddev = Input(shape=[1], name="stddev")
    resnet_conf = Input(shape=[1], name="resnet_conf")
    xception_conf = Input(shape=[1], name="xception_conf")
    inception_conf = Input(shape=[1], name="inception_conf")
    #'resnet_conf', 'xception_conf', 'inception_conf'
    
    #Embeddings layers
    
    
    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title_description)
    #emb_seq_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_description)
    #emb_seq_title = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title)
    
    emb_region = Embedding(max_region, emb_size)(region)
    emb_city = Embedding(max_city, emb_size)(city)
    emb_category_name = Embedding(max_category_name, emb_size)(category_name)
    emb_parent_category_name = Embedding(max_parent_category_name, emb_size)(parent_category_name)
    emb_param_1 = Embedding(max_param_1, emb_size)(param_1)
    emb_param123 = Embedding(max_param123, emb_size)(param123)
    emb_image_code = Embedding(max_image_code, emb_size)(image_code)

    #rnn_layer1 = GRU(gru_size) (emb_seq_title_description)
    #rnn_layer1 = GRU(gru_size, dropout=0.1,recurrent_dropout=0.1) (emb_seq_title_description)
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
    
    rnn_layer1 = CuDNNGRU(gru_size)(emb_seq_title_description)
    #rnn_layer1 = CuDNNGRU(gru_size)(emb_seq_description)
    #rnn_layer2 = CuDNNGRU(gru_size)(emb_seq_title)
    
    """
    rnn_layer1 = Bidirectional(CuDNNGRU(gru_size, return_sequences=True))(emb_seq_title_description)
    avg_tensor = GlobalAveragePooling1D()(rnn_layer1)
    max_tensor = GlobalMaxPooling1D()(rnn_layer1)
    rnn_layer1 = Concatenate()([avg_tensor, max_tensor])
    """
    #main layer
    main_l = concatenate([
          rnn_layer1
        #, rnn_layer2
        , Flatten() (emb_region)
        , Flatten() (emb_city)
        , Flatten() (emb_category_name)
        , Flatten() (emb_parent_category_name)
        , Flatten() (emb_param_1)
        , Flatten() (emb_param123)
        , Flatten() (emb_image_code)
        , avg_days_up_user
        , avg_times_up_user
        , n_user_items
        , price
        , item_seq_number
        , whratio
        , laplacian
        , colorfull
        , brightness
        , median
        , rms
        , stddev
        , resnet_conf
        , xception_conf
        , inception_conf
    ])
        
    #main_l = Dropout(0.1)(Dense(512,activation='relu') (main_l))
    #main_l = Dropout(0.1)(Dense(64,activation='relu') (main_l))
    
    
    main_l = BatchNormalization()(main_l)
    #main_l = Dropout(0.20)(main_l)
    main_l = Dense(512)(main_l)
    main_l = PReLU()(main_l)
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.2)(main_l)
    main_l = Dense(64)(main_l)
    main_l = PReLU()(main_l)
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.1)(main_l)
    
    
    #output
    output = Dense(1,activation="sigmoid") (main_l)
    
    #model
    model = Model([seq_title_description, 
                   #seq_description, seq_title, 
                   region, city, category_name, 
                   parent_category_name, param_1, param123, price, 
                   item_seq_number, image_code, 
                   avg_days_up_user, avg_times_up_user, n_user_items,
                   whratio, laplacian, colorfull, brightness, median, rms, stddev, resnet_conf,xception_conf,inception_conf
                   ], output)
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
        for df in pd.read_csv('../input/test.csv', chunksize= 3000000):
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
        
        img_att = []
        test, imgchanged = calcImgAtt(test, img_att)
        test.drop('image', axis=1, inplace=True)
        
        #print(test.dtypes)
        
        test['avg_days_up_user'] = test['avg_days_up_user'].fillna(0).astype('uint32')
        test['avg_times_up_user'] = test['avg_times_up_user'].fillna(0).astype('uint32')
        test['n_user_items'] = test['n_user_items'].fillna(0).astype('uint32')
        test = keras_test_transform(test)
        del df
        gc.collect()
    
        #print(test.dtypes)
    
        X_test = get_keras_data(test)
        del test 
        gc.collect()
    
        Batch_Size = 512*3
        preds1 = model.predict(X_test, batch_size = Batch_Size, verbose = 1)
        #print(preds1.shape)
        del X_test
        gc.collect()
        print("RNN Prediction is done")

        preds1 = preds1.reshape(-1,1)
        #print(predsl.shape)
        preds1 = np.clip(preds1, 0, 1)
        #print(preds1.shape)
        item_ids = np.append(item_ids, item_id)
        #print(item_ids.shape)
        preds = np.append(preds, preds1)
        #print(preds.shape)
        
    print("All chunks done")
    t2 = time.time()
    print("Total time for Parallel Batch Prediction is "+str(t2-t1))
    return preds 

# Loading Train data - No Params, No Image data 
dtypes_train = {
                'price': 'float32',
                'deal probability': 'float32',
                'item_seq_number': 'uint32'
}

# No user_id
use_cols = ['item_id', 'user_id', 'image_top_1', 'region', 'city', 
            'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 
            'description', 'price', 'item_seq_number', 'activation_date', 'deal_probability',
            'image']
train = pd.read_csv("../input/train.csv", skiprows=range(1,frm), nrows=to-frm, parse_dates=["activation_date"], usecols = use_cols, dtype = dtypes_train)

train_features = pd.read_csv('../input/aggregated_features.csv')
train = train.merge(train_features, on = ['user_id'], how = 'left')
del train_features
gc.collect()

img_att = []
train, imgchanged = calcImgAtt(train, img_att)
train.drop('image', axis=1, inplace=True)

emb_cols = ['region', 'city', 'category_name', 'parent_category_name', 
            'param_1', 'param123', 'image_code']

train['avg_days_up_user'] = train['avg_days_up_user'].fillna(0).astype('uint32')
train['avg_times_up_user'] = train['avg_times_up_user'].fillna(0).astype('uint32')
train['n_user_items'] = train['n_user_items'].fillna(0).astype('uint32')

y_train = np.array(train['deal_probability'])

del train['deal_probability']
gc.collect()


train = preprocess_dataset(train)
train, tokenizer, dict_encoder = keras_fit(train)
train = keras_train_transform(train)
print("Tokenization done and TRAIN READY FOR Validation splitting")

# Calculation of max values for Categorical fields 

max_region = np.max(train.region.max())+2
max_city= np.max(train.city.max())+2
max_category_name = np.max(train.category_name.max())+2
max_parent_category_name = np.max(train.parent_category_name.max())+2
max_param_1 = np.max(train.param_1.max())+2
max_param123 = np.max(train.param123.max())+2
#max_week = np.max(train.week.max())+2
#max_day = np.max(train.day.max())+2
#max_wday = np.max(train.wday.max())+2
max_image_code = np.max(train.image_code.max())+2


del train['item_id'], train['user_id']
gc.collect()


EMBEDDING_FILE1 = '../input/wiki.ru.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

vocab_size = len(tokenizer.word_index)+2
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


"""
EMBEDDING_FILE1 = '../input/avito_emb.w2v'
model = Word2Vec.load(EMBEDDING_FILE1)
word_index = tokenizer.word_index
vocab_size = min(max_words_title_description, len(word_index))
embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
for word, i in word_index.items():
    if i >= max_words_title_description: continue
    try:
        embedding_vector = model[word]
    except KeyError:
        embedding_vector = None
    if embedding_vector is not None: embedding_matrix1[i] = embedding_vector

print(" FAST TEXT DONE")
"""



from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time 
skf = KFold(n_splits = 5)
Kfold_preds_final = []
k = 0
RMSE = []

cnt = 0
for train_idx, test_idx in skf.split(train.values, y_train):
    
    print("Number of Folds.."+str(k+1))
    
    
    #K Fold Split 
    
    X_train1, X_test1 = train.iloc[train_idx], train.iloc[test_idx]
    #print(X_train1.shape, X_test1.shape)
    y_train1, y_test1 = y_train[train_idx], y_train[test_idx]
    #print(y_train1.shape, y_test1.shape)
    gc.collect()
    
    X_train_final = get_keras_data(X_train1)
    X_test_final = get_keras_data(X_test1)
    #del X_train1, X_test1
    #gc.collect
    
    
    # Initialize a new Model for Current FOLD 
    epochs = 1
    batch_size = 512 * 3
    steps = (int(train.shape[0]/batch_size))*epochs
    lr_init, lr_fin = 0.009, 0.0045
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    modelRNN = RNN_model()
    K.set_value(modelRNN.optimizer.lr, lr_init)
    K.set_value(modelRNN.optimizer.decay, lr_decay)

    # Fit the NN Model 
    
    #for i in range(5):
        #hist = modelRNN.fit(X_train_final, y_train1, batch_size=batch_size+(batch_size*(2*i)), epochs=epochs, validation_data=(X_test_final, y_test1), verbose=1)
        #hist = modelRNN.fit(X_train_final, y_train1, batch_size=batch_size+(batch_size*i), epochs=epochs, validation_data=(X_test_final, y_test1), verbose=1)
    
    
    file_path = "../model/bestrnn_{}.hdf5".format(cnt)
    cnt += 1
    check_point = ModelCheckpoint(file_path, monitor='val_root_mean_squared_error', mode='min', 
                                  save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_root_mean_squared_error', patience=2, mode='min')
    hist = modelRNN.fit(X_train_final, y_train1, batch_size=batch_size, epochs=500, 
                        validation_data=(X_test_final, y_test1), verbose=1,
                        callbacks=[check_point, early_stop])
    
    
    
    del X_train_final
    gc.collect()
    
    from keras import backend as K
    K.clear_session()

    
    modelRNN = RNN_model()
    K.set_value(modelRNN.optimizer.lr, lr_init)
    K.set_value(modelRNN.optimizer.decay, lr_decay)
    modelRNN.load_weights(file_path)
    
    # Print RMSE for Validation set for Kth Fold 
    v_rmse = eval_model(modelRNN, X_test_final)
    RMSE.append(v_rmse)
    
    del X_test_final
    del y_train1, y_test1
    gc.collect()
    
    # Predict test set for Kth Fold 
    preds = predictions(modelRNN)
    del modelRNN 
    gc.collect()

    print("Predictions done for Fold "+str(k))
    print(preds.shape)
    Kfold_preds_final.append(preds)
    del preds
    gc.collect()
    print("Number of folds completed...."+str(len(Kfold_preds_final)))
    #print(Kfold_preds_final[k][0:10])
    k += 1
    
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
test = pd.read_csv('../input/test.csv', usecols = test_cols)

# using Average of KFOLD preds 

submission1 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission1['item_id'] = test['item_id']
submission1['deal_probability'] = pred_final1

#print("Check Submission NOW!!!!!!!!@")
#submission1.to_csv("Avito_Shanth_RNN_AVERAGE.csv", index=False)

import kaggle_util
kaggle_util.save_result(submission1, '../result/rnn_avg.csv', competition = 'avito-demand-prediction', send = not debug, index = False)

"""
# Using KFOLD preds with Minimum value 
submission2 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission2['item_id'] = test['item_id']
submission2['deal_probability'] = pred_final2

print("Check Submission NOW!!!!!!!!@")
submission2.to_csv("Avito_Shanth_RNN_MIN.csv", index=False)
"""