#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:58:18 2018

@author: kownse
"""

import os
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

def resize_dir(path, outpath, IMG_SIZE = 128):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    root, dirs, files = next(os.walk(path))
    for file in tqdm(files):
        fpath = os.path.join(path, file)
        savepath = os.path.join(outpath, file)
        
        if os.path.exists(savepath):
            continue
        
        try:
            img = imread(fpath)
        except:
            continue
        
        img = resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
        
        imsave(savepath, img)
        
if __name__ == "__main__":
    #pass
    #resize_dir('../input/test_jpg/data/competition_files/test_jpg/', '../input/test_jpg_128/')
    resize_dir('../input/train_jpg/data/competition_files/train_jpg/', '../input/train_jpg_128/')