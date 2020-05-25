import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
from keras import models,layers
from keras.models import Model,model_from_json,Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D, Add
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class XceptionModel:

    def __init__(self,dict):
        self.filters = dict['filters']
        self.filtersize = dict['filtersize']
        self.layers = dict['layers']

    def NNgen(self,inputs):
        a = self.filters
        b = self.filtersize
        x = keras.layers.concatenate([SeparableConv2D(a,b[0],strides=1,padding='same')(inputs),SeparableConv2D(a,b[1],strides=1,padding='same')(inputs)])
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        previous_activation = x

        for i in range(0,self.layers):

            x = keras.layers.concatenate([SeparableConv2D(a,b[0],strides=1,padding='same')(x),SeparableConv2D(a,b[1],strides=1,padding='same')(x)])
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = Add()([x,previous_activation])
            previous_activation = x

        x = Conv2D(1,(1,1),activation='relu',padding='same')(x)

        return x
