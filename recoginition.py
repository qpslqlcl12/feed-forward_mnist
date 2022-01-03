import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

#data=([])
label = tf.Variable([0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.])

input = tf.Variable(tf.zeros(shape=(1,5), dtype=tf.float32))
print(input)
conc=tf.concat([label,input], -1)
print(conc)
w1=tf.Variable(tf.random.normal([5,2],stddev=0.35, dtype=tf.float32), name='w1')
print(w1)
h1=tf.matmul(input,w1  )
print(h1)
h1=tf.keras.activations.sigmoid(h1)
print(h1)
