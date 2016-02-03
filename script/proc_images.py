#!/usr/bin/env python3

import numpy as np
import os
import scipy.ndimage as spi
import scipy.misc as spm
import multiprocessing
import pickle
import pandas as pd

DATA = 'data/'
TRAINING = DATA+'train_photos/'
N_IMGS = 1000

def proc_img(fname):
    path = TRAINING+str(fname)[0]+'/'+str(fname)+'.jpg'
    return(spm.imresize(spi.imread(path),(128,128)))

p = multiprocessing.Pool(multiprocessing.cpu_count())

labels = pd.read_csv(DATA+'train.csv')
imgmap = pd.read_csv(DATA+'train_photo_to_biz_ids.csv')
labels = labels.merge(imgmap, on='business_id')

X = np.zeros((N_IMGS,128,128,3))
y = labels[:N_IMGS]['labels'].str.split(r'\s+', expand=True).

futures = list()

for i in labels[:N_IMGS].to_records():
    futures.append(p.apply_async(proc_img, (i[3],)))

for i, future in enumerate(futures):
    X[i,...] = future.get()

print(X.shape)
print(y.shape)
print(y)
np.save(DATA+'X.npy', X)
np.save(DATA+'y.npy', y)
