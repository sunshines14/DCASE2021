import os
import sys
import keras
import numpy as np
from keras.optimizers import SGD
from utils import *
from generator import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#=========================================================================================================#
data_path = #path
train_csv = #csvfile
val_csv = data_path + #csvfile

train_feat_path = #path
valid_feat_path = #path

experiments = #path
if not os.path.exists(experiments):
    os.makedirs(experiments)

#=========================================================================================================#
num_audio_channels = 1
num_freq_bin = 128
num_time_bin = 423
num_classes = 10
max_lr = 0.1
batch_size = 32
num_epochs = 256
mixup_alpha = 0.4
sample_num = len(open(train_csv, 'r').readlines()) - 1

model_selection = 0
focal_loss = False
use_split = False

#=========================================================================================================#    
data_val, y_val = load_data_2020(valid_feat_path, val_csv, num_freq_bin, 'logmel')
data_deltas_val = deltas(data_val)
data_deltas_deltas_val = deltas(data_deltas_val)
data_val = np.concatenate((data_val[:,:,4:-4,:], data_deltas_val[:,:,2:-2,:], data_deltas_deltas_val), axis=-1)
y_val_onehot = keras.utils.to_categorical(y_val, num_classes)

#=========================================================================================================#
num_audio_channels = 3*num_audio_channels
if model_selection == 0:
    from models.mobnet_ca import model_mobnet_ca
    model = model_mobnet_ca(num_classes, 
                            input_shape = [num_freq_bin, num_time_bin, num_audio_channels], 
                            num_filters = 32,
                            wd = 1e-3,
                            use_split = use_split)
    
if model_selection == 1:
    from models.mobnet_fusion import model_mobnet_fusion
    model = model_mobnet_fusion(num_classes, 
                                input_shape = [num_freq_bin, num_time_bin, num_audio_channels], 
                                num_filters = 32,
                                wd = 1e-3,
                                use_split = use_split)
    
model.summary()
print (data_val.shape)

#=========================================================================================================#
model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(lr=max_lr, decay=1e-6, momentum=0.9, nesterov=False),
              metrics = ['accuracy'])

#=========================================================================================================#
lr_scheduler = LR_WarmRestart(nbatch = np.ceil(sample_num/batch_size), 
                              Tmult = 2,
                              initial_lr = max_lr,
                              min_lr = max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0])

save_path = experiments + "/model-{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, 
                                             monitor = 'val_loss', 
                                             verbose = 1, 
                                             save_best_only = True, 
                                             mode = 'min')
callbacks = [lr_scheduler, checkpoint]

#=========================================================================================================#
train_data_generator = Generator_timefreqmask_withdelta_nocropping_splitted(train_feat_path,
                                                                            train_csv, 
                                                                            num_freq_bin,
                                                                            batch_size = batch_size,
                                                                            alpha = mixup_alpha,
                                                                            splitted_num = 20)()

history = model.fit(train_data_generator,
                    validation_data = (data_val, y_val_onehot),
                    epochs = num_epochs, 
                    verbose = 1, 
                    workers = 8,
                    max_queue_size = 100,
                    callbacks = callbacks,
                    steps_per_epoch = np.ceil(sample_num/batch_size)) 
