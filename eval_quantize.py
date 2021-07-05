import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from plots import plot_confusion_matrix

#=========================================================================================================#
def evaluate_model(interpreter, test_images, test_labels, num_class, is_eval=False):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    prediction_digits = []
    pred_output_all = np.empty([1, num_class])
    for test_image in test_images:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_index)
        pred_output = output[0]
        pred_output.reshape([1, num_class])
        pred_output_all = np.vstack((pred_output_all, pred_output))
        digit = np.argmax(output[0])
        prediction_digits.append(digit)
             
    pred_output_all = pred_output_all[1:,:]
    
    if is_eval:
        return pred_output_all, prediction_digits
    else:
        accurate_count = 0
        for index in range(len(prediction_digits)):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)
        return accuracy, pred_output_all, prediction_digits

#=========================================================================================================#
is_eval = False

if not is_eval:
    data_path = #path
    val_csv = data_path + #csvfile
    feat_path = #path
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    
else:
    data_path = #path
    val_csv = data_path + #csvfile
    feat_path = #path
    model_path = sys.argv[1]
    csv_path = sys.argv[2].replace('.csv','-eval.csv')

num_freq_bin = 128
num_classes = 10

print (model_path)
print (csv_path)

#=========================================================================================================#
if not is_eval:
    data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
    data_deltas_val = deltas(data_val)
    data_deltas_deltas_val = deltas(data_deltas_val)
    data_val = np.concatenate((data_val[:,:,4:-4,:], data_deltas_val[:,:,2:-2,:], data_deltas_deltas_val), axis=-1)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
    print(data_val.shape)
    print(y_val.shape)

    dev_test_df = pd.read_csv(val_csv, sep='\t', encoding='ASCII')
    wav_paths = dev_test_df['filename'].tolist()
    
    device_idxs = []
    for idx, elem in enumerate(wav_paths):
        wav_paths[idx] = wav_paths[idx].split('/')[-1]
        device_idxs.append(wav_paths[idx].split('.')[0].split('-')[-1])
    device_list = np.unique(device_idxs)
    
    class_idxs = []
    for idx, elem in enumerate(wav_paths):
        wav_paths[idx] = wav_paths[idx].split('/')[-1]
        class_idxs.append(wav_paths[idx].split('.')[0].split('-')[0])
    class_list = np.unique(class_idxs)
    
else:
    data_val = load_data_2020_evaluate(feat_path, val_csv, num_freq_bin, 'logmel')
    data_deltas_val = deltas(data_val)
    data_deltas_deltas_val = deltas(data_deltas_val)
    data_val = np.concatenate((data_val[:,:,4:-4,:], data_deltas_val[:,:,2:-2,:], data_deltas_deltas_val), axis=-1)
    print(data_val.shape)
    
    dev_test_df = pd.read_csv(val_csv, sep='\t', encoding='ASCII')
    wav_paths = dev_test_df['filename'].tolist()
    
    for idx, elem in enumerate(wav_paths):
        wav_paths[idx] = wav_paths[idx].split('/')[-1]
    
#=========================================================================================================#
interpreter_quant = tf.lite.Interpreter(model_path=model_path)
interpreter_quant.allocate_tensors()

#=========================================================================================================#

if not is_eval:
    overall_acc, preds, preds_class_idx = evaluate_model(interpreter_quant, 
                                                         data_val, 
                                                         y_val, 
                                                         num_class=num_classes,
                                                         is_eval=False)

    over_loss = log_loss(y_val_onehot, preds)
    print("\n\nVal acc: ", "{0:.4f}".format(overall_acc))
    print("Val log loss: ", "{0:.4f}".format(over_loss))

    device_acc = []
    device_loss = []
    for device_id in device_list:
        cur_preds = np.array([preds[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id])
        cur_y_pred_val = np.argmax(cur_preds,axis=1)
        cur_y_val_onehot = np.array([y_val_onehot[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id])
        cur_y_val = [y_val[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id]
        cur_loss = log_loss(cur_y_val_onehot, cur_preds)
        cur_acc = np.sum(cur_y_pred_val==cur_y_val) / len(cur_preds)
    
        device_acc.append(cur_acc)
        device_loss.append(cur_loss)
    
    print("\n\nDevices list: ", device_list)
    print("Per-device val acc : ", np.array(device_acc))
    print("Per-device val loss : ", np.array(device_loss))

    # get confusion matrix
    y_pred_val = np.argmax(preds, axis=1)
    conf_matrix = confusion_matrix(y_val, y_pred_val)
    plot_confusion_matrix(y_val, y_pred_val, class_list, normalize=True, title=None, png_name=csv_path.replace('.csv','.png'))
    print("\n\nConfusion matrix:")
    print(conf_matrix)
    
    class_acc = []
    class_loss = []
    for class_id in class_list:
        cur_preds = np.array([preds[i] for i in range(len(class_idxs)) if class_idxs[i] == class_id])
        cur_y_pred_val = np.argmax(cur_preds, axis=1)
        cur_y_val_onehot = np.array([y_val_onehot[i] for i in range(len(class_idxs)) if class_idxs[i] == class_id])
        cur_y_val = [y_val[i] for i in range(len(class_idxs)) if class_idxs[i] == class_id]
        cur_loss = log_loss(cur_y_val_onehot, cur_preds)
        cur_acc = np.sum(cur_y_pred_val==cur_y_val) / len(cur_preds)
    
        class_acc.append(cur_acc)
        class_loss.append(cur_loss)
    
    print("\n\nclasses list: ", class_list)
    print("Per-class val acc : ", np.array(class_acc))
    print("Per-class val loss : ", np.array(class_loss))

else:
    preds, preds_class_idx = evaluate_model(interpreter_quant, 
                                            data_val, 
                                            test_labels=None, 
                                            num_class=num_classes,
                                            is_eval=True)
    y_pred_val = np.argmax(preds, axis=1)

#=========================================================================================================#
scene_map_str = """
airport 0 
bus 1
metro 2
metro_station 3
park 4
public_square 5
shopping_mall 6
street_pedestrian 7
street_traffic 8
tram 9
"""

scene_index_map={}
for line in scene_map_str.strip().split('\n'):
    ch, index = line.split()
    scene_index_map[int(index)] = ch
labels = [str(scene_index_map[c]) for c in y_pred_val]
filename = [str(a[:]) for a in wav_paths]
left = {'filename': filename, 'scene_label': labels}
left_df = pd.DataFrame(left)
right_df = pd.DataFrame(preds, columns = ['airport',
                                          'bus',
                                          'metro',
                                          'metro_station',
                                          'park',
                                          'public_square',
                                          'shopping_mall',
                                          'street_pedestrian',
                                          'street_traffic',
                                          'tram'] )
merge = pd.concat([left_df, right_df], axis=1, sort=False)
merge.to_csv(csv_path, sep = '\t', index=False)
