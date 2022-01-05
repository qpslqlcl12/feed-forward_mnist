import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

np.set_printoptions(precision=6, suppress=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

#print("xtrain", x_train[0])
#print("ytrain", y_train[0])
data_5=x_train[0]
#print(data_5.shape)
which_label=y_train[0]
print("label:", which_label) #5
#def for label data generator
label = tf.Variable([0.,0.,0.,0.,0.,1.,0.,0.,0.,0.])
actual_value=label
#input = tf.Variable(tf.zeros(shape=(5), dtype=tf.float32))
#print("input", input)

#def data_preprocessing():


def network_initializer(label, data_5):
    conc=tf.concat([label,data_5], 0)
    #print("conc shape", conc.shape)
    dims=tf.expand_dims(conc,0)
    #print("inputs:",dims)
    input=dims
    w1=tf.Variable(tf.random.uniform([794,400],-1,1), trainable=True, name='w1')
    #print("weight",w1)    
    w2=tf.Variable(tf.random.uniform([400,10],-1,1), trainable=True, name='w2')
    return input, w1, w2

loss_fn=tf.keras.losses.MeanSquaredError()
opt=tf.keras.optimizers.Adam(learning_rate=0.01)


def feed_forward(input,w1,w2):
    h1=tf.matmul(input,w1)    
    h1=tf.keras.activations.sigmoid(h1)
    #print(h1)
    out=tf.matmul(h1,w2)
    out=tf.keras.activations.sigmoid(out)
    return out,w1,w2

def loss(input,w1,w2,actual_value):
    prediction=feed_forward(input,w1,w2)
    loss_value=loss_fn(actual_value, prediction)
    return loss_value


def grad(input,w1,w2):
    with tf.GradientTape() as tape:
        loss_value=loss(input,w1,w2,actual_value)   
    dw1, dw2 = tape.gradient(loss_value,[w1,w2])
    return loss_value,dw1,dw2
 
    
#sequence
input, w1, w2 = network_initializer(label, data_5)


for i in range(10):
    prediction,w_buffer1,w_buffer2 = feed_forward(input, w1, w2)
    print(i,"th iteration")
    print("prediction: ", prediction)    
    print("w_buffer1:", w_buffer1)
    print("w_buffer2:", w_buffer2)
    #training
    loss_value, grad_weight1, grad_weight2=grad(input, w1,w2)
    print("loss_value:", loss_value)
    opt.apply_gradients(zip([grad_weight1,grad_weight2],[w1,w2]))



