import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, MaxPool2D, MaxPool3D,UpSampling2D,UpSampling3D,
    Reshape,ZeroPadding2D,ZeroPadding3D,
)
from tensorflow.keras.layers import Concatenate,Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization

import Defaults

conv_dict = {2: Conv2D, 3: Conv3D}
max_pooling_dict = {2: MaxPool2D, 3: MaxPool3D}
up_sampling_dict = {2: UpSampling2D, 3: UpSampling3D}
zero_sampling_dict = {2: ZeroPadding2D, 3: ZeroPadding3D}

cg = Defaults.Parameters()



def conv_bn_relu_1x(nb_filter, kernel_size, subsample, dimension, weight_decay):
    sub = subsample * dimension
    Conv = conv_dict[dimension]

    def f(input_layer):
        conv_a = Conv(
            filters=nb_filter,
            kernel_size=kernel_size,
            strides=sub,
            padding="same",
            use_bias=False,
            kernel_initializer="orthogonal",
            kernel_regularizer=l2(weight_decay),
            bias_regularizer=l2(weight_decay),
        )(input_layer)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation("relu")(norm_a)
        return act_a

    return f

def conv_bn_relu(
    nb_filter, kernel_size, subsample=(1,), dimension=2, repeat=2, weight_decay=1e-4
):
    def f(input_layer):
        inputs = [input_layer]
        for i in range(repeat):
            act = conv_bn_relu_1x(
                nb_filter, kernel_size, subsample, dimension, weight_decay
            )(inputs[i])
            inputs += [act]
        return inputs[repeat]

    return f


def get_CNN(dim,conv_depth,layer_name=None,dimension=2,weight_decay=1e-4,unet_depth=5):
    # conv_depth = [16,32,64,128,256,128,64,32,16] by default

    # define model parameters
    if np.isscalar(dim):
        dim = (dim,) * dimension
    else:
        assert len(dim) == dimension

    assert len(conv_depth) == 2 * unet_depth - 1

    kernel_size = (3,) * dimension
    pool_size = (2,) * dimension
    MaxPooling = max_pooling_dict[dimension]
    UpSampling = up_sampling_dict[dimension]
    ZeroPadding = zero_sampling_dict[dimension]
    Conv = conv_dict[dimension]

    def f(input_layer):
        levels = []
        ds = []
        us = []

        # downsampling path
        for i in range(0,unet_depth):
            if i == 0:
                levels += [conv_bn_relu(conv_depth[i], kernel_size, dimension=dimension)(input_layer)]
                
            else:
                levels += [conv_bn_relu(conv_depth[i], kernel_size, dimension=dimension)(ds[-1])]
            
            if i <= unet_depth -2:
                ds += [MaxPooling(pool_size=pool_size)(levels[-1])]

        # for i in range(len(levels)):
        #     print(i, ' th levels, shape: ', levels[i].shape)

        # for i in range(len(ds)):
        #     print(i, ' th ds, shape: ', ds[i].shape)

            
        # upsampling path
        for i in range(0,unet_depth - 2):
            up_layer = UpSampling(size = pool_size)(levels[-1])
            # print(i, 'th up layer, shape: ',up_layer.shape)
            # print(' concatenate with levels with shape: ',levels[unet_depth - 2 - i].shape)
            
            us += [Concatenate(axis = -1)([up_layer,levels[unet_depth - 2 - i]])]
            # print(us[-1].shape)

            levels += [conv_bn_relu(conv_depth[i + unet_depth], kernel_size, dimension=dimension)(us[-1])]

        # final
        up_layer = UpSampling(size = pool_size)(levels[-1])
        us += [Concatenate(axis = -1)([up_layer,levels[0]])]
        levels += [conv_bn_relu_1x(conv_depth[-1], kernel_size, (1,), dimension, weight_decay)(us[-1])]

        final_feature = conv_bn_relu_1x(1, kernel_size, (1,), dimension, weight_decay)(levels[-1])

        final_image = Conv(filters=1,kernel_size=(1,1),padding = "same", name = layer_name)(final_feature) # do we need reshape here?
        print('final layer dimension: ',final_image.shape)

        return levels, ds, us, final_feature,final_image

    return f ## important


def learning_rate_step_decay(epoch, lr, step=cg.lr_epochs, initial_power=cg.initial_power):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    num = epoch // step
    lrate = 10 ** (initial_power - num)
    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)




        



 
