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
import gc

class UNetTallModel:
    def __init__(self, dict):

        self.filters = dict['filters']
        self.depth = dict['depth']

    def transposecalc(self,inputdim, outputdim):
        # if padding='same', then outputdim = inputdim * stride
        # if padding='valid', then outputdim = (inputdim - 1]*stride + filtersize
        if float(outputdim) % float(inputdim) == 0:
            newfiltersize = int(0.5*(float(outputdim)/4.0)+0.5*(float(inputdim)/4.0))+1
            newstride = int(outputdim/inputdim)
            return [newfiltersize,newfiltersize], newstride, 'same'
        else:
            newstride = int(float(outputdim)/float(inputdim-1))
            newfiltersize = outputdim - (inputdim-1)*newstride
            return [newfiltersize,newfiltersize], newstride, 'valid'


    def create_model(self, inputs):

        convlist = []
        x = inputs
        print(" UNetTall model with SeparableConv2D")
        print(" input for second flow shape ", x.get_shape().as_list())

        for i in range(0, self.depth):
            print(" contraction layer at ", i )
            print("    before    ", x.get_shape().as_list())
            a = self.filters
            b1 = int(float(x.get_shape().as_list()[1])/float(8.0))
            b2 = int(float(x.get_shape().as_list()[1]) / float(4.0))
            print(" filter size " , a, '  ', b1, '   ', b2)
            x = tf.keras.layers.concatenate([Conv2D(a,b1,strides=1,padding='same')(x),Conv2D(a,b2,strides=1,padding='same')(x)])
            x = Activation(activation='relu')(x)
            x = BatchNormalization()(x)
            convlist.append(x)
            x = Dropout(0.25)(x)
            print("    after    ", x.get_shape().as_list())



        print(" before middle flow shape " , x.get_shape().as_list())
        b1 = int(float(x.get_shape().as_list()[1])/float(8.0))
        b2 = int(float(x.get_shape().as_list()[1])/float(4.0))
        print(" filter size ",'  ', b1, '   ', b2)
        x = tf.keras.layers.concatenate([Conv2D(self.filters,b1,strides=1,padding='same')(x),Conv2D(a,b2,strides=1,padding='same')(x)])
        print(" after middel flow shape " , x.get_shape().as_list())
        x = Activation(activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)


        for i in range(0,self.depth):

            print(" expansion layer at ", i )
            print("    before    ", x.get_shape().as_list())
            a = int(float(self.filters)/float(2.0))
            b1 = int(float(x.get_shape().as_list()[1])/float(8.0))
            b2 = int(float(x.get_shape().as_list()[1]) / float(4.0))
            print(" filter size " , a, '  ', b1, '   ', b2)
            x = tf.keras.layers.concatenate([x,convlist[self.depth-1-i]])
            print("    after concate ", x.get_shape().as_list())
            x = tf.keras.layers.concatenate([Conv2D(a,b1,strides=1,padding='same',)(x),Conv2D(a,b2,strides=1,padding='same')(x)])
            x = Activation(activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
        x = Conv2D(1,(1,1),activation='relu',padding='same')(x)
        print(" output shape ", x.get_shape().as_list())
        convlist = []

        return x
    def NNgen(self,inputs):
        x = self.create_model(inputs)
        return x
