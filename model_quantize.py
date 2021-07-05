import tensorflow as tf
import sys

hdf5_path = sys.argv[1]
tflite_path = sys.argv[2]

model = tf.keras.models.load_model(hdf5_path)
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open(tflite_path, "wb") as output_file:
    output_file.write(tflite_quant_model)
