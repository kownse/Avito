#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:28:23 2018

@author: kownse
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc
from keras import backend as K
from keras.layers import Dense,Input,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from capsule import Capsule

import kaggle_util
from profiler import profile
# Any results you write to the current directory are saved as output.

max_features=20000
maxlen=100
embed_size=300

EMBEDDING_FILE = '../input/wiki.ru.vec'

def getModel():
    Routings = 6
    Num_capsule = 10
    Dim_capsule = 16
    rate_drop_dense = 0.35
    
    #def root_mean_squared_error(y_true, y_pred):
    #    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    sequence_input = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(32, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                          share_weights=True)(x)
    capsule = Flatten()(capsule)
    capsule = Dropout(rate_drop_dense)(capsule)
    preds = Dense(1, activation="sigmoid")(capsule)
    model = Model(sequence_input, preds)
    model.compile(loss='MSE',optimizer=Adam(lr=1e-3),metrics=['accuracy', root_mean_squared_error])
    
    return model

@profile
def loadEmbbeding_Index():
    embeddings_index = {}
    with open(EMBEDDING_FILE,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float16')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings_index = loadEmbbeding_Index()


@profile
def loadData():
    train = pd.read_csv('../input/train.csv', usecols=['description', 'deal_probability'])
    test = pd.read_csv('../input/test.csv', usecols=['description'])
    
    train['description'] = train['description'].astype(str)
    test["description"] = test['description'].astype(str)
    train["description"].fillna("fillna")
    test["description"].fillna("fillna")
    X_train = train["description"]
    y_train = train[["deal_probability"]].values
    del train
    gc.collect()
    
    X_test = test["description"]
    del test
    gc.collect()

    
    from keras.preprocessing import text, sequence
    tok=text.Tokenizer(num_words=max_features)
    tok.fit_on_texts(X_train)
    X_train=tok.texts_to_sequences(X_train)
    X_test=tok.texts_to_sequences(X_test)
    x_train=sequence.pad_sequences(X_train,maxlen=maxlen)
    del X_train
    gc.collect()
    x_test=sequence.pad_sequences(X_test,maxlen=maxlen)
    del X_test
    gc.collect()

            
    word_index = tok.word_index
    #prepare embedding matrix
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    del tok
    gc.collect()
    
    
    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)
    del x_train
    gc.collect()
    del y_train
    gc.collect()
    
    return X_tra, X_val, y_tra, y_val, x_test

if __name__ == "__main__":
    
    
    X_tra, X_val, y_tra, y_val, x_test = loadData()
    
    
    model = getModel()
    batch_size = 3000
    epochs = 10
    # filepath="../input/best-model/best.hdf5"
    filepath="../model/weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_root_mean_squared_error', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_root_mean_squared_error", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    
    #model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)
    #Loading model weights
    model.load_weights(filepath)
    print('Predicting....')
    y_pred = model.predict(x_test,batch_size=1024,verbose=1)
    
    sub = pd.read_csv('../input/sample_submission.csv')
    sub['deal_probability'] = y_pred
    sub['deal_probability'].clip(0.0, 1.0, inplace=True)
    sub.to_csv('gru_capsule_description.csv', index=False)
    
    kaggle_util.save_result(sub, '../result/capsule.csv', competition = 'avito-demand-prediction', send = True)