#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:20:17 2018

@author: kownse
"""

import os
import copy
import pandas as pd
from gensim.models import Word2Vec
from random import shuffle
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import logging
import time
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
use_cols = ['param_1','param_2','param_3','title', 'description']


def load_text(start):
    print('Loading data...', end='')
    tic = time.time()
    train2 = pd.read_csv('../input/train_active.csv', usecols=use_cols, nrows= 1000000, skiprows=range(1, start))
    toc = time.time()
    print('Done in {:.1f}s'.format(toc-tic))
    train2['text'] = train2['param_1'].str.cat([train2.param_2,train2.param_3,train2.title,train2.description], sep=' ',na_rep='')
    train2.drop(use_cols, axis = 1, inplace=True)
    train2 = train2['text'].values

    train2 = [text_to_word_sequence(text) for text in tqdm(train2)]
    return train2

model = Word2Vec(size=300, window=5,max_vocab_size=500000)

for k in range(15):
    update = False
    if k != 0:
        update = True
    train = load_text(k*1000000+1)
    model.build_vocab(train, update=update)
    model.train(train, total_examples=model.corpus_count, epochs=3)

"""
train1 = pd.read_csv('../input/train.csv', usecols=use_cols)
train2 = pd.read_csv('../input/train_active.csv', usecols=use_cols)
trainall = train1.append(train2).dropna(subset=['description'])

del train1, train2
trainall['text'] = trainall['param_1'].str.cat([trainall.param_2,trainall.param_3,trainall.title,trainall.description], sep=' ',na_rep='')
trainall.drop(use_cols, axis = 1, inplace=True)
trainall = trainall['text'].values
trainall = [text_to_word_sequence(text) for text in tqdm(trainall)]
model = Word2Vec(size=300, window=5,max_vocab_size=500000)
model.build_vocab(trainall, update=True)
model.train(trainall, total_examples=model.corpus_count, epochs=3)
"""
 
model.save('../input/avito_emb.w2v')