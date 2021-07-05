import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *


def se_channel_attention(x, reduction_ratio=8):
    ch_input = K.int_shape(x)[-1]
    ch_reduced = ch_input//reduction_ratio
    
    y = GlobalAveragePooling2D()(x)
    
    y = Dense(ch_reduced, kernel_initializer='he_normal', activation='relu', use_bias=False)(y)
    y = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(y)
    
    y = tf.expand_dims(y, 1)
    y = tf.expand_dims(y, 1)
    
    y = x * y
    return y


def coordinate_attention(x, reduction_ratio=8):
    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x
    
    x_shape = K.int_shape(x)
    [b, h, w, c] = x_shape
    
    x_h = AveragePooling2D(pool_size=(1, w), strides=1, padding='valid')(x)
    x_w = AveragePooling2D(pool_size=(h, 1), strides=1, padding='valid')(x)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    
    mip = max(8, c // reduction_ratio)
    y = Conv2D(mip, (1, 1), strides=1, padding='valid')(y)
    
    y = BatchNormalization(axis=-1)(y)
    y = coord_act(y)
    
    x_h, x_w = tf.split(y, num_or_size_splits=[h, w], axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    
    a_h = Conv2D(c, (1, 1), strides=1, padding='same')(x_h)
    a_h = Activation('sigmoid')(a_h)
    a_w = Conv2D(c, (1, 1), strides=1, padding='same')(x_w)
    a_w = Activation('sigmoid')(a_w)
    
    out = x * a_h * a_w
    return out
