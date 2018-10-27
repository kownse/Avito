#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:46:30 2018

@author: kownse
"""

import pickle
import os
import cv2
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
from PIL import Image, ImageStat
from io import BytesIO
from multiprocessing import Pool
from keras.preprocessing import image as kimage
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
import pandas as pd

def save(dic, file):
    with open(file, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(file):
    with open(file, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
 
	# compute rg = R - G
	rg = np.absolute(R - G)
 
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
 
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
 
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
 
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

def check_add_dict(dict_att, key, att, value):
    if att not in dict_att:
        dict_att[att] = {}
    dict_att[att][key] = value
    
def image_classify(model, pak, img, top_n=1):
    """Classify image and return top matches."""
    """
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = kimage.img_to_array(img)
    """
    x = img.copy()
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=top_n)[0]

def calcAttEach(key, fpath):
    try:
        img = imread(fpath)
    
        pil_image = Image.open(fpath).convert('L')
        stat = ImageStat.Stat(pil_image)
    
        width = img.shape[0]
        height = img.shape[1]

        check_add_dict(dict_att, key, 'whratio', width / height)
        check_add_dict(dict_att, key, 'area', np.log(width * height + 0.001))
        check_add_dict(dict_att, key, 'laplacian', cv2.Laplacian(img, cv2.CV_64F).var())
        check_add_dict(dict_att, key, 'colorfull', image_colorfulness(img))
        
        check_add_dict(dict_att, key, 'brightness', stat.mean[0])
        check_add_dict(dict_att, key, 'median', stat.median[0])
        check_add_dict(dict_att, key, 'rms', stat.rms[0])
        check_add_dict(dict_att, key, 'stddev', stat.stddev[0])
        
    except:
        print('can not open img ' + fpath)
        
def calcConfidenceEach(key, fpath, df_conf):
    try:
        img = imread(fpath)
        target_size = (224, 224)
        if img.shape[:2] != target_size:
            img = resize(img, target_size, preserve_range=True)
        resnet_preds = image_classify(resnet_model, resnet50, img)
        xception_preds = image_classify(xception_model, xception, img)
        inception_preds = image_classify(inception_model, inception_v3, img)
        
        resnet_conf = resnet_preds[0][2]
        xception_conf = xception_preds[0][2]
        inception_conf = inception_preds[0][2]
        
        check_add_dict(dict_att, key, 'resnet_conf', resnet_conf)
        check_add_dict(dict_att, key, 'xception_conf', xception_conf)
        check_add_dict(dict_att, key, 'inception_conf', inception_conf)
        
        df_conf = df_conf.append({'resnet_conf': resnet_conf, 
                                  'xception_conf': xception_conf, 
                                  'inception_conf': inception_conf}, ignore_index=True)
        
    except:
        print('can not open img ' + fpath)
        
    return df_conf

def calcImgLabelEach(key, fpath):
    try:
        img = imread(fpath)
        target_size = (224, 224)
        if img.shape[:2] != target_size:
            img = resize(img, target_size, preserve_range=True)
        inception_preds = image_classify(inception_model, inception_v3, img)
        
        inception_conf = inception_preds[0][2]
        inception_label = inception_preds[0][1]
        if inception_conf > 0.3:
            check_add_dict(dict_att, key, 'img_label', inception_label)
            #print('conf label {}[{}]'.format(inception_label, inception_conf))
        
    except:
        print('can not open img ' + fpath)
        
    return df_conf
    

def calcTextureAtt(path, df_conf):
    
    root, dirs, files = next(os.walk(path))
    
    cnt = 0
    for file in tqdm(files):    
        key = os.path.splitext(os.path.basename(file))[0]
        fpath = os.path.join(path, file)
        
        #calcAttEach(key, fpath)
        #df_conf = calcConfidenceEach(key, fpath, df_conf)
        calcImgLabelEach(key, fpath)
        """
        cnt += 1
        if cnt > 1:
            break
        """
    print('save')
    #df_conf.to_csv('../input/img_conf_score.csv')
    save(dict_att, dict_att_path)
        
if __name__ == "__main__":
        
    dict_att_path = '../input/dict_imgatt.pkl'
    
    print('load model')
    #resnet_model = resnet50.ResNet50(weights='imagenet')
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    #xception_model = xception.Xception(weights='imagenet')
    print('model loaded')
    #path = '../input/test_jpg/data/competition_files/test_jpg/'
    #path = '../input/train_jpg/data/competition_files/train_jpg/'


    dict_att = {}
    if os.path.exists(dict_att_path):
        dict_att = load(dict_att_path)
    
    df_conf = pd.DataFrame(columns=['resnet_conf', 'xception_conf', 'inception_conf'])
    calcTextureAtt('../input/train_jpg_128/', df_conf)
    calcTextureAtt('../input/test_jpg_128/', df_conf)
    
    #df_score = pd.read_csv('../input/img_conf_score.csv')