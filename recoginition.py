import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

#data=([])
input = tf.Variable(tf.zeros(shape=(50), dtype=tf.float32))

w1=tf.Variable(tf.random.normal([50], dtype=tf.float32), name='w1')

h1=tf.matmul(input, w1)
print(h1)
