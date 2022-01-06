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
#print("label:", which_label) #5
#def for label data generator
label = tf.Variable([0.,0.,0.,0.,0.,1.,0.,0.,0.,0.])
actual_value=label
actual_value=tf.expand_dims(actual_value,0)
#input = tf.Variable(tf.zeros(shape=(5), dtype=tf.float32))
#print("input", input)

#def data_preprocessing():

w1=[]
w2=[]
encoder_buffer=[]
TCAM_buffer=[]
reverse_prediction=[]
TCAM_array=[]

loss_fn=tf.keras.losses.MeanSquaredError()
opt=tf.keras.optimizers.Adam(learning_rate=0.01)

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

def forward_pass(input,w1,w2):
    h1=tf.matmul(input,w1)    
    h1=tf.keras.activations.sigmoid(h1)
    #print(h1)
    out=tf.matmul(h1,w2)
    out=tf.keras.activations.sigmoid(out)
    return out,w1,w2

def backward_pass(encoder_buffer,w1,w2):
    w2=tf.transpose(w2)
    h1=tf.matmul(encoder_buffer,w2)
    h1=tf.keras.activations.sigmoid(h1)
    w1=tf.transpose(w1)
    input=tf.matmul(h1,w1)
    input=tf.keras.activations.sigmoid(input)
    return input

def loss(input,w1,w2,actual_value):
    prediction,dummy1,dummy2=forward_pass(input,w1,w2)
    
    #print("pred", prediction)
    #print("actual", actual_value)
    
    loss_value=loss_fn(actual_value, prediction)
    return loss_value

def grad(input,w1,w2):
    with tf.GradientTape() as tape:
        loss_value=loss(input,w1,w2,actual_value)   
    dw1, dw2 = tape.gradient(loss_value,[w1,w2])   
    return loss_value,dw1,dw2
 
def TCAM_store(encoded_data):
    TCAM_array.append(encoded_data)
    return 0
  
def TCAM_retrieve(encoded_query_data):
    dist1=[]
    dist2=[100.] 
    #print("here:", TCAM_array)
    for stored_data in TCAM_array:
        #print("stored:",stored_data)
        dist1.append(Minkowski_distance(encoded_query_data, stored_data))
    #print("dist1:", dist1)
    min_dist=dist1.index(min(dist1))
    #print("min_dist: ", min_dist )
        #if dist1 < dist2 :
            #dist2=dist1
            #print("muyaho", dist2)           
    #print("min_TCAM_array:", TCAM_array[min_dist])
    #dist1=Minkowski_distance(encoded_query_data, TCAM_array)
    #print("current_dist:", dist1)
    
    return TCAM_array[min_dist], min_dist

def Minkowski_distance(x,y):
    #print(x)
    #print(y)
    dist=tf.sqrt(tf.reduce_sum(tf.square(x-y)))
    #print("distance: ", dist)
    return dist
#sequence
input, w1, w2 = network_initializer(label, data_5)
#print("input:", input)

for i in range(5):
    #inference
    prediction,w1,w2 = forward_pass(input, w1, w2)
    #print(i,"th iteration")
    #print("prediction: ", prediction)    
    
    #encoder_buffer=prediction
    #reverse_prediction=backward_pass(encoder_buffer,w1,w2)
    #print("reverse prediction:", reverse_prediction)
    #print("label prediction:", tf.slice(reverse_prediction,[0,0],[1,10]))
    loss_value, grad_weight1, grad_weight2=grad(input, w1,w2)
    print("loss_value:", loss_value)
    opt.apply_gradients(zip([grad_weight1,grad_weight2],[w1,w2]))
   
    TCAM_store(prediction)
    #print("TCAM stored: ",TCAM_array)
    
print("TCAM data:", TCAM_array)
#print("dist:",Minkowski_distance(prediction,label))
prediction,w1,w2 = forward_pass(input, w1, w2)
print("query_data: ", prediction)
TCAM_buffer, distance=TCAM_retrieve(prediction)
print("retrieved_data: ", TCAM_buffer)
print("distance:", distance)