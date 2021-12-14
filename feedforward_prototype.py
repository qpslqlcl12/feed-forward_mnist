import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

numepoch = 5
numtrainimages = 1000

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255


inputs = keras.Input(shape=(784,))
dense = layers.Dense(784, activation="sigmoid")
x = dense(inputs)
x = layers.Dense(400, activation="sigmoid")(x)
outputs = layers.Dense(10)(x)
        

for epochs in range(1, numepoch):
    for batchsize in range(1, numtrainimages): 
       
    
    
    
    
    for nHidden in range(1, epoch):
        for nInput in range()


