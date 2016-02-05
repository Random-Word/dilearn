#!/usr/bin/env python

import numpy as np
import os
import cv2
#import scipy.ndimage as spi
#import scipy.misc as spm
import multiprocessing
import pandas as pd
import re

DATA = 'data/'
TRAINING = DATA+'train_photos/'
N_IMGS = 10000

mean_pixel = np.zeros((1, 1, 1, 3))
mean_pixel[0,0,0,:] = [103.939, 116.779, 123.68]

def index2arr(s):
    to_ret = np.zeros((8))
    s = re.split(r'\s+', s)
    for i in s:
        to_ret[int(i)-1] = 1
    return to_ret

def proc_img(fname):
    path = TRAINING+str(fname)[0]+'/'+str(fname)+'.jpg'
    return(cv2.resize(cv2.imread(path),(128,128)))

p = multiprocessing.Pool(multiprocessing.cpu_count())

labels = pd.read_csv(DATA+'train.csv')
imgmap = pd.read_csv(DATA+'train_photo_to_biz_ids.csv')
labels = labels.merge(imgmap, on='business_id')

X = np.zeros((N_IMGS,128,128,3))
y = np.vstack(labels[:N_IMGS]['labels'].apply(index2arr).values)

futures = list()

for i in labels[:N_IMGS].to_records():
    futures.append(p.apply_async(proc_img, (i[3],)))

for i, future in enumerate(futures):
    X[i,...] = future.get()

X -= mean_pixel
print(X.shape)
print(y.shape)
print(y)
np.save(DATA+'X.npy', X)
np.save(DATA+'y.npy', y)
