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

from xceptionnet import XceptionModel

outputblocksize = 5
outputblockindex = 0


def data_gen(dataset, blocksize, blockindex):

    batch = dataset[:,:,:,:-1]

    label = []

    for i in range(0,len(dataset)):
        temp = []
        for j in range(blocksize * blockindex, blocksize * blockindex + blocksize):
            for q in range(blocksize * blockindex, blocksize * blockindex + blocksize):
                temp.append(dataset[i][j][q][14])
        label.append(temp)

    return batch, label


parameters = {"firstConv_filters": 100, "firstConv_filterSize": [2, 2], "firstConv_filterStride": 1,
              "entry_residual_blocks": 10, "entry_residual_filters": 200, "entry_residual_filterSize": [2,2],
              "entry_residual_filterStride": 1, "fully_connected_flow_layers": 5}


validx = int(0.9 * 75000)

trainset = data_gen(np.load('../train.npy')[0:validx], outputblocksize, outputblockindex)

valset = data_gen(np.load('../train.npy')[validx:], outputblocksize, outputblockindex)



model = XceptionModel(parameters)

width, height, depth = 40, 40, 14  # hyperparameter

inputs = Input(shape=(width, height, depth))

outputs = model.forward(inputs, outputblocksize)

xception = Model(inputs, outputs)

xception.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

batch_size = 100

epochs = 150



''' the lines below can be used for trainig if train and test data is prepared '''


# lookback , rough avg of # of filters, rough avg of filter size, # of layers, epoch
checkpoint = keras.callbacks.ModelCheckpoint('model_'+str(outputblockindex)+'_'+'{epoch:08d}'+'.h5',period=10)
history = xception.fit(trainset[0], trainset[1], callbacks=[checkpoint],epochs=epochs, batch_size=batch_size,
                       validation_data=(valset[0], valset[1]))





