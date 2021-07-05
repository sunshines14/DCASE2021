import os
import pickle
import random
import numpy as np
import pandas as pd
import threading
import keras
from keras import backend as K


def load_data_2020(feat_path, csv_path, feat_dim, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        lines = lines[1:]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info = np.array(lines)
        
        data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
        ClassNames = np.unique(data_df['scene_label'])
        labels = data_df['scene_label'].astype('category').cat.codes.values

        feat_mtx = []
        for [filename, labnel] in label_info:
            filepath = feat_path + '/' + filename + '.logmel' 
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)
        return feat_mtx, labels


def load_data_2020_splitted(feat_path, csv_path, feat_dim, idxlines, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        lines = lines[1:]
        lines = [lines[i] for i in idxlines]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info = np.array(lines)
        
        data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
        ClassNames = np.unique(data_df['scene_label'])
        labels = data_df['scene_label'].astype('category').cat.codes.values
        labels = [labels[i] for i in idxlines]

        feat_mtx = []
        for [filename, label] in label_info:
            filepath = feat_path + '/' + filename + '.' + 'logmel'
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)
        return feat_mtx, labels

def load_data_2020_evaluate(feat_path, csv_path, feat_dim, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        lines = lines[1:]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        info = np.array(lines)
        
        data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
        feat_mtx = []
        for [filename] in info:
            filepath = feat_path + '/' + filename + '.logmel' 
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)
        return feat_mtx


def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out
    
    
def frequency_masking(mel_spectrogram, frequency_masking_para=13, frequency_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)
        
        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0+f),:] = 0
    return mel_spectrogram


def time_masking(mel_spectrogram, time_masking_para=40, time_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0+t)] = 0
    return mel_spectrogram


def cmvn(data):
    shape = data.shape
    eps = 2**-30
    for i in range(shape[0]):
        utt = data[i].squeeze().T
        mean = np.mean(utt, axis=0)
        utt = utt - mean
        std = np.std(utt, axis=0)
        utt = utt / (std + eps)
        utt = utt.T
        data[i] = utt.reshape((utt.shape[0], utt.shape[1], 1))
    return data


def frequency_label(num_sample, num_frequencybins, num_timebins):
    data = np.arange(num_frequencybins, dtype='float32').reshape(num_frequencybins, 1) / num_frequencybins
    data = np.broadcast_to(data, (num_frequencybins, num_timebins))
    data = np.broadcast_to(data, (num_sample, num_frequencybins, num_timebins))
    data = np.expand_dims(data, -1)
    return data


class LR_WarmRestart(keras.callbacks.Callback):
    def __init__(self,nbatch,initial_lr,min_lr,epochs_restart,Tmult):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs_restart = epochs_restart
        self.nbatch = nbatch
        self.currentEP=0
        self.startEP=0
        self.Tmult=Tmult
        
    def on_epoch_begin(self, epoch, logs={}):
        if epoch+1<self.epochs_restart[0]:
            self.currentEP = epoch
        else:
            self.currentEP = epoch+1
            
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            self.Tmult=2*self.Tmult
        
    def on_epoch_end(self, epochs, logs={}):
        lr = K.get_value(self.model.optimizer.lr)
        print ('\nLearningRate:{:.6f}'.format(lr))
    
    def on_batch_begin(self, batch, logs={}):
        pts = self.currentEP + batch/self.nbatch - self.startEP
        decay = 1+np.cos(pts/self.Tmult*np.pi)
        lr = self.min_lr+0.5*(self.initial_lr-self.min_lr)*decay
        K.set_value(self.model.optimizer.lr,lr)

        
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
        

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g
