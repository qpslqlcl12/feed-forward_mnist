import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

#print("xtrain", x_train[0])
#print("ytrain", y_train[0])
data_5=x_train[0]
#print(data_5.shape)
which_label=y_train[0]
print(which_label) #5
#def for label data generator
label = tf.Variable([0.,0.,0.,0.,0.,1.,0.,0.,0.,0.])
actual_value=label
#input = tf.Variable(tf.zeros(shape=(5), dtype=tf.float32))
#print("input", input)


def network_initializer():
    conc=tf.concat([label,data_5], 0)
#print("conc shape", conc.shape)
    dims=tf.expand_dims(conc,0)
#print(dims)
    input=dims
    w1=tf.Variable(tf.random.uniform([794,400],0,1), trainable=True, name='w1')
#print("weight",w1)    
    w2=tf.Variable(tf.random.uniform([400,10],0,1), trainable=True, name='w2')
    return input, w1, w2

loss_fn=tf.keras.losses.MeanSquaredError()
opt=tf.keras.optimizers.Adam(lr=0.01)
losses = []

def feed_forward(input,w1,w2):
    h1=tf.matmul(input,w1)    
    h1=tf.keras.activations.sigmoid(h1)
    print(h1)
    out=tf.matmul(h1,w2)
    out=tf.keras.activations.sigmoid(out)
    return out
"""
def loss(input,w1,w2,actual_value):
    prediction=feed_forward(input,w1,w2)
    loss_value=loss_fn(actual_value, prediction)
    return loss_value


def grad(input,w1,w2):
    with tf.GradentTape() as tape:
        loss_value=loss(input,w1,w2,actual_value)
        dw1, dw2 = tape.gradient(loss,[w1,w2])
    return dw1,dw2
         
#sequence
input, w1, w2 = network_initializer()


prediction = feed_forward(input, w1, w2)

grad_weight1, grad_weight2=grad(input, w1,w2)
opt.apply_gradients(zip([grad_weight1,grad_weight2],[w1,w2]))

losses.append(loss(w1,w2))
print("loss: "+str(loss(w1,w2).numpy()))
"""