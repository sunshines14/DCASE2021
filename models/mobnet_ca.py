import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from models.attention import se_channel_attention, coordinate_attention


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def freq_split1(x):
    return x[:, 0:64, :, :]


def freq_split2(x):
    return x[:, 64:128, :, :]


def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation('relu')(x)


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)  
    return x


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    cchannel = int(filters * alpha)
    
    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])
    return x


def mobile_net_block(inputs, first_filters, alpha):
    x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))  
    x = _inverted_residual_block(x, 32, (3, 3), t=2, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 48, (3, 3), t=2, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=2, alpha=alpha, strides=2, n=3)
    
    if alpha > 1.0:
        last_filters = _make_divisible(64 * alpha, 8)
    else:
    	last_filters = 64

    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
    return x


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, learn_bn=True, wd=1e-4, use_relu=True):
    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    
    if use_relu:
        x = Activation('relu')(x)
        
    x = Conv2D(num_filters,
               kernel_size = kernel_size,
               strides = strides,
               padding = 'same',
               kernel_initializer = 'he_normal',
               kernel_regularizer = l2(wd),
               use_bias = False)(x)
    return x


def model_mobnet_ca(num_classes, input_shape=[128, None, 3], num_filters=32, wd=1e-3, alpha=1, use_split=False):
    inputs = Input(shape=input_shape)
    first_filters = _make_divisible(num_filters * alpha, 8)
    
    if use_split:
        Split1 = Lambda(freq_split1)(inputs)
        Split2 = Lambda(freq_split2)(inputs)
        Split1 = mobile_net_block(Split1, first_filters, alpha)
        Split2 = mobile_net_block(Split2, first_filters, alpha)
        MobilePath = concatenate([Split1, Split2], axis=1)
    else:
        MobilePath = mobile_net_block(inputs, first_filters, alpha)

    OutputPath = resnet_layer(inputs = MobilePath,
                              num_filters = num_filters * 2,
                              kernel_size = 1,
                              strides = 1,
                              learn_bn = False,
                              wd = wd,
                              use_relu = True)
    
    OutputPath = Dropout(0.3)(OutputPath)
    OutputPath = coordinate_attention(OutputPath, reduction_ratio=4) #penultimate
    
    OutputPath = resnet_layer(inputs = OutputPath,
                              num_filters = num_classes,
                              strides = 1,
                              kernel_size = 1,
                              learn_bn = False,
                              wd = wd,
                              use_relu = False)
    
    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)

    model = Model(inputs=inputs, outputs=OutputPath)
    return model