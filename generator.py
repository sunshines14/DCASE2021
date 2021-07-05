import sys
import pandas
import numpy as np
import threading
import keras
from keras import backend as K
from utils import *
sys.path.append("..")


class Generator_timefreqmask_withdelta_nocropping_splitted():
    def __init__(self, feat_path, train_csv, feat_dim, batch_size=32, alpha=0.2, shuffle=True, splitted_num=4): 
        self.feat_path = feat_path
        self.train_csv = train_csv
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(open(train_csv, 'r').readlines())-1
        self.lock = threading.Lock()
        self.splitted_num = splitted_num
        
    def __iter__(self):
        return self
    
    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()

                item_num = self.sample_num // self.splitted_num - (self.sample_num // self.splitted_num) % self.batch_size
                for k in range(self.splitted_num):
                    cur_item_num = item_num
                    s = k * item_num
                    e = (k+1) * item_num 
                    if k == self.splitted_num - 1:
                        cur_item_num = self.sample_num - (self.splitted_num - 1) * item_num
                        e = self.sample_num

                    lines = indexes[s:e]
                    X_train, y_train = load_data_2020_splitted(self.feat_path, self.train_csv, self.feat_dim, lines, 'logmel')
                    y_train = keras.utils.to_categorical(y_train, 10)
                    X_deltas_train = deltas(X_train)
                    X_deltas_deltas_train = deltas(X_deltas_train)
                    X_train = np.concatenate((X_train[:,:,4:-4,:], X_deltas_train[:,:,2:-2,:], X_deltas_deltas_train), axis=-1)
                    
                    itr_num = int(cur_item_num // (self.batch_size * 2))
                    
                    for i in range(itr_num):
                        batch_ids = np.arange(cur_item_num)[i*self.batch_size * 2:(i + 1) * self.batch_size * 2]
                        X, y = self.__data_generation(batch_ids, X_train, y_train)
                        yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids, X_train, y_train):
        _, h, w, c = X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = X_train[batch_ids[:self.batch_size]]
        X2 = X_train[batch_ids[self.batch_size:]]
        
        for j in range(X1.shape[0]):
            for c in range(X1.shape[3]):
                X1[j, :, :, c] = frequency_masking(X1[j, :, :, c])
                X1[j, :, :, c] = time_masking(X1[j, :, :, c])
                X2[j, :, :, c] = frequency_masking(X2[j, :, :, c])
                X2[j, :, :, c] = time_masking(X2[j, :, :, c])
                
        X = X1 * X_l + X2 * (1.0 - X_l)

        if isinstance(y_train, list):
            y = []

            for y_train_ in y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = y_train[batch_ids[:self.batch_size]]
            y2 = y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1.0 - y_l)

        return X, y
