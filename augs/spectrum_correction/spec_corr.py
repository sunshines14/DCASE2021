import os
import random
import pickle
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sound
from sklearn.preprocessing import StandardScaler


#=========================================================================================================#
data_path = #path
train_csv_file = data_path + #csvfile
device_a_csv = data_path + #csvfile

output_path = #path
feature_type = 'logmel'

if not os.path.exists(output_path):
    os.makedirs(output_path)

#=========================================================================================================#
sr = 44100
num_audio_channels = 1
num_channel = 1
duration = 10

num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))

#=========================================================================================================#
dev_train_df = pd.read_csv(train_csv_file, sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()

trainf = [x[6:-4] for x in wavpaths_train]
train_subset = []

for idx in ['-a', '-b', '-c']:
    train_subset.append([x[:-1] for x in trainf if (x.endswith(idx))])

for idx in ['-s1', '-s2', '-s3', '-s4', '-s5', '-s6']:
    train_subset.append([x[:-2] for x in trainf if (x.endswith(idx))])

train_sets=[]
for idx in range(len(train_subset)):
    train_sets.append(set(train_subset[idx]))

#=========================================================================================================#
paired_wavs = []
for j in range(1, len(train_sets)):
    paired_wavs.append(train_sets[0] &  train_sets[j])

num_paired_wav = [len(x) for x in paired_wavs]
min_paired_wav = 150

waves30 = []
wav_idxs = random.sample(range(min(num_paired_wav)), min_paired_wav)
for wavs in paired_wavs:
    temp = [list(wavs)[i] for i in wav_idxs]
    waves30.append(temp)

#=========================================================================================================#
nbins_stft = int(np.ceil(num_fft/2.0)+1)
STFT_all = np.zeros((len(waves30)*min_paired_wav, nbins_stft, num_time_bin), 'float32')

for group, x in zip(waves30, ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']):
    i = 0
    for sc in group:
        wav_a = 'audio/' + sc + 'a.wav'
        wav_x = 'audio/' + sc + x + '.wav'
        stereo_a, fs = sound.read(data_path+wav_a, stop=duration*sr)
        stereo_x, fs = sound.read(data_path+wav_x, stop=duration*sr)

        STFT_a = librosa.stft(stereo_a, n_fft=num_fft, hop_length=hop_length)
        STFT_x = librosa.stft(stereo_x, n_fft=num_fft, hop_length=hop_length)

        STFT_ref = np.abs(STFT_x)
        STFT_corr_coeff = STFT_ref/np.abs(STFT_a)

        STFT_all[i,:,:] = STFT_corr_coeff
        i=i+1

STFT_hstak = np.hstack(STFT_all)
STFT_corr_coeff = np.expand_dims(np.mean(STFT_hstak, axis=1), -1)

data_df = pd.read_csv(device_a_csv, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()

#=========================================================================================================#
for d in range(1):
    for i in range(len(wavpath)):
        stereo, fs = sound.read(data_path + wavpath[i], stop=duration*sr)
        STFT = librosa.stft(stereo, n_fft=num_fft, hop_length=hop_length)
        STFT_corr = np.abs(STFT)*STFT_corr_coeff
        
        logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
        logmel_data[:,:,0] = librosa.feature.melspectrogram(S=np.abs(STFT_corr)**2, 
                                                            sr=sr, 
                                                            n_fft=num_fft, 
                                                            hop_length=hop_length, 
                                                            n_mels=num_freq_bin, 
                                                            fmin=0.0, 
                                                            fmax=sr/2, 
                                                            htk=True, 
                                                            norm=None)
        logmel_data = np.log(logmel_data+1e-8)
        
        for j in range(len(logmel_data[:,:,0][:,0])):
            mean = np.mean(logmel_data[:,:,0][j,:])
            std = np.std(logmel_data[:,:,0][j,:])
            logmel_data[:,:,0][j,:] = ((logmel_data[:,:,0][j,:]-mean)/std)
            logmel_data[:,:,0][np.isnan(logmel_data[:,:,0])]=0.

        feature_data = {'feat_data': logmel_data}
        cur_file_name = output_path + wavpath[i][5:-4] + '-speccorr.'  + feature_type
        pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print (cur_file_name)
