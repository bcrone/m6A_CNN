import fastparquet
import numpy as np
import pandas as pd
import os
import sys
import h5py
import tqdm
import pickle
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.stats as ss
from datetime import datetime
from keras.models import load_model
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import gc

def onehot_encode(seq):
    mapping = pd.Series(index = ["A", "C", "G", "T"], data = [0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

# Load full kmer set
print('Load full kmer set')
data = pd.read_parquet('/path/to/validation.parquet.gzip')
data = data[data['seq'].apply(lambda x: len(x) == 13)].reset_index(drop=True)
array_counts = data.loc[:,"count"].values.reshape(-1,1)
target = data.loc[:,"prob"].values
seqs = data.loc[:, "seq"].values
del data
gc.collect()

# One-hot encoding of kmer sequences
print('One-hot encoding of kmer sequences')
n_jobs = 10
with mp.Pool(n_jobs) as pool:
    onehot_seqs = np.stack(list(tqdm.tqdm(pool.imap(onehot_encode, seqs, chunksize=5000), total=len(seqs))))
del seqs
gc.collect()

val_data = onehot_seqs
val_counts = array_counts
val_target = target

model = load_model("/path/to/model.h5")

val_pred = np.transpose(model.predict([val_data, val_counts]))[0]
_, mse_loss = model.evaluate([val_data, val_counts], val_target)
print("MSE on validation set " + str(mse_loss))
print("Pred-target correlation " + str(ss.pearsonr(val_target, val_pred)[0]))