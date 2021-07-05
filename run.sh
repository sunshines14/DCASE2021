#!/bin/bash

hdf5_path=#path
tflite_path=#path
save_path=#path

python3 -W ignore model_quantize.py $hdf5_path $tflite_path
python3 -W ignore model_size.py $hdf5_path $tflite_path
python3 -W ignore eval_quantize.py $tflite_path $save_path
