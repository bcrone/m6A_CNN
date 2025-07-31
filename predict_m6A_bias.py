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
data = pd.read_parquet('/nfs/turbo/boylelab/crone/projects/m6A/data/hg38.analysisSet.aligned.filtered.sorted.m6A.13mer.binned_chrAll.summary.parquet.gzip')
#data['seq'] = data.index.astype('str')
data = data[data['seq'].apply(lambda x: len(x) == 13)].reset_index(drop=True)
#data['sum'] = data['sum'].astype(np.float32)
#data['mean'] = data['mean'].astype(np.float32)
#data = data[data['sum'] > 1].reset_index(drop=True)
print(f"Number of candidate kmers: {data.shape[0]}")
#array_counts = data.loc[:,"sum"].values.reshape(-1,1)
#target = data.loc[:,"mean"].values
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

# Generate training/testing/validation partitions
print('Generate training/testing/validation partitions')
np.random.seed(42)
inds = np.arange(target.shape[0])
np.random.shuffle(inds)
training_inds = inds[:int(target.shape[0]* 0.8)]
val_inds = inds[int(target.shape[0] * 0.8):int(target.shape[0] * 0.9)]
test_inds = inds[int(target.shape[0] * 0.9):]

# Split the encoded sequences and target values into training, validation and test
training_data = onehot_seqs[training_inds]
training_counts = array_counts[training_inds]
training_target = target[training_inds]
val_data = onehot_seqs[val_inds]
val_counts = array_counts[val_inds]
val_target = target[val_inds]
test_data = onehot_seqs[test_inds]
test_counts = array_counts[test_inds]
test_target = target[test_inds]

# Include reverse complements of sequences
print('Include reverse complements of sequences')
training_reverse_complement = np.flip(np.flip(training_data, axis = 1), axis = 2)
training_data = np.concatenate([training_data, training_reverse_complement])
training_counts = np.concatenate([training_counts, training_counts])
training_target = np.concatenate([training_target, training_target])
np.shape(training_data)
inds = np.arange(np.shape(training_data)[0])
np.random.shuffle(inds)
training_data = training_data[inds]
training_counts = training_counts[inds]
training_target = training_target[inds]

# Construct model
seq_input = Input(shape = (np.shape(test_data[0])))
freq_input = Input(shape = (np.shape(test_counts[0])))
conv_1 = Conv1D(32, 5, padding = 'same', activation = 'relu', strides = 1)(seq_input)
maxpool_1 = MaxPooling1D()(conv_1)
conv_2 = Conv1D(32, 5, padding = 'same', activation = 'relu', strides = 1)(maxpool_1)
maxpool_2 = MaxPooling1D()(conv_2)
conv_3 = Conv1D(32, 5, padding = 'same', activation = 'relu', strides = 1)(maxpool_2)
maxpool_3 = MaxPooling1D()(conv_3)
flat = Flatten()(maxpool_3)
dense = Dense(8, activation = "relu")(freq_input)
combined = concatenate([flat,dense])
fc = Dense(64,activation = "relu")(combined)
out = Dense(1,activation = "linear")(fc)
model = Model(inputs=[seq_input, freq_input],outputs=out)  
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

prev_loss = np.inf
for n_epoch in range(10):

    # New training epoch
    model.fit([training_data, training_counts], training_target, batch_size=64, epochs = 1, 
                validation_data=([val_data, val_counts], val_target))  

    # Get MSE loss on the valdation set after current epoch
    val_pred = np.transpose(model.predict([val_data, val_counts]))[0]
    _, mse_loss = model.evaluate([val_data, val_counts], val_target)
    print("MSE on validation set " + str(mse_loss))

    # Get pred-target correlation on the validation set after current epoch
    print("Pred-target correlation " + str(ss.pearsonr(val_target, val_pred)[0]))

    # If loss on validation set starts to increase, stop training and adopt the previous saved version
    if mse_loss > prev_loss:
        break
    else:
        prev_loss = mse_loss
        # Save current model version
        model.save("/nfs/turbo/boylelab/crone/projects/m6A/data/m6A_NN_model.13mer.binned.h5")