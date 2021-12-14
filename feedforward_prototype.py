import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import time

numepoch = 5
numtrainimages = 1000

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
       

inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")
outputs = layers.Dense(10, name="predictions")(x1)

print(inputs)   #tensorshape (none, 784)

"""
for epochs in range(1, numepoch):
    for batchsize in range(1, numtrainimages):
        
       
    
    
    
    
    for nHidden in range(1, epoch):
        for nInput in range()
"""

