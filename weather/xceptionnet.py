import numpy as np
import math
import time
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import os
import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, \
    UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class XceptionModel:
    def __init__(self, dict):
        self.firstConv_filters = dict["firstConv_filters"]  # number of filters
        self.firstConv_filterSize = dict["firstConv_filterSize"]  # size of filters
        self.firstConv_filterStride = dict["firstConv_filterStride"]  # stride of filters

        self.entry_residual_blocks = dict["entry_residual_blocks"]
        self.entry_residual_filters = dict["entry_residual_filters"]
        self.entry_residual_filterSize = dict["entry_residual_filterSize"]
        self.entry_residual_filterStride = dict["entry_residual_filterStride"]

        self.fully_connected_flow_layers = dict["fully_connected_flow_layers"]

    def entry_flow(self, inputs):
        # entry convolutional layers

        x = SeparableConv2D(self.firstConv_filters, self.firstConv_filterSize,
                   strides=self.firstConv_filterStride, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('selu')(x)

        previous_block_activation = x

        print(" first conv layer   ", previous_block_activation.get_shape().as_list())

        for _ in range(self.entry_residual_blocks):
            print(" residual block at ", _, "   ", x.get_shape().as_list())
            x = Activation('selu')(x)
            x = SeparableConv2D(self.entry_residual_filters, self.entry_residual_filterSize,
                       strides=self.entry_residual_filterStride, padding='same')(x)
            x = BatchNormalization()(x)

            # max pooling layer that we may potentially get rid of
            x = MaxPooling2D(3, strides=2, padding='same')(x)

            # the residual connection as described in the architecture diagram
            residual = SeparableConv2D(self.entry_residual_filters, 1, strides=2, padding='same')(previous_block_activation)
            x = Add()([x, residual])
            previous_block_activation = x

#        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        return x

    def fully_connected_res_flow(self, x):
        num_nodes = x.get_shape().as_list()
        temp = Dense(num_nodes[1])(x)
        temp = Activation(activation="selu")(temp)
        temp = BatchNormalization()(temp)
        temp = Add()([x, temp])

        return temp

    def fully_connected_flow(self, x, outputsize):
        print(" shape after all conv layers " , x.get_shape().as_list())

        for _ in range(self.fully_connected_flow_layers):
            temp = self.fully_connected_res_flow(x)
            x = temp
            print(" fully residual block at " , _ , "  ", x.get_shape().as_list())

        x = Dense(outputsize, activation='linear')(x)

        print(" final shape " , x.get_shape().as_list())

        return x

    def forward(self, input, outputsize):
        x = self.entry_flow(input)
        x = self.fully_connected_flow(x, outputsize)
        return x


