import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import random

np.set_printoptions(precision=6, suppress=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
"""
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
"""

training_label_list=[0,1,2,3,4,5,6,7,8,9]
#training_label_list.append(random.randint(0,9))
training_label_list=random.sample(training_label_list,4)
print("random",training_label_list)
#rad=np.random.choice(y_train,4,replace=False)
#print("rad",rad)


support_data_set=[]
for j in training_label_list:
    while len(support_data_set) <=4:
        random_label=random.randint(0,59999)        
        if j == y_train[random_label]:
            print("randomlabel",random_label)
            support_data_set.append(x_train[random_label])
            print("label:", y_train[random_label])
       
print("support_data_set", len(support_data_set))


#def support_set_sampling():
    

#support data set
data_0=x_train[1]
data_1=x_train[3]
data_2=x_train[5]
data_3=x_train[7]
data_4=x_train[2]
data_5=x_train[0]
data_6=x_train[13]
data_8=x_train[17]
data_9=x_train[4]

#query data set
Qdata_0=x_train[21]
Qdata_1=x_train[6]
Qdata_2=x_train[16]
Qdata_3=x_train[10]
Qdata_4=x_train[9]
Qdata_5=x_train[11]
Qdata_6=x_train[18]
Qdata_7=x_train[15]
Qdata_8=x_train[31]
Qdata_9=x_train[19]

#label for 4 way 
way_0 = tf.Variable([1.,0.,0.,0.])
way_1 = tf.Variable([0.,1.,0.,0.])
way_2 = tf.Variable([0.,0.,1.,0.])
way_3 = tf.Variable([0.,0.,0.,1.])

empty_label = tf.Variable([0.,0.,0.,0.])

w1=[]
w2=[]
w3=[]
encoder_buffer=[]
TCAM_buffer=[]
#reverse_prediction=[]
TCAM_array=[]
backward_output=[]
support_data_set=[]
query_input=[]

opt=tf.keras.optimizers.Adam(learning_rate=0.01)
def data_preprocessing(way):
    
    data_00=tf.concat([way_0,data_0],0)
    data_11=tf.concat([way_1,data_1],0)
    data_33=tf.concat([way_2,data_3],0)
    data_55=tf.concat([way_3,data_5],0)

    data_00=tf.expand_dims(data_00,0)
    data_11=tf.expand_dims(data_11,0)
    data_33=tf.expand_dims(data_33,0)
    data_55=tf.expand_dims(data_55,0)    

    support_data_set.append(data_00)
    support_data_set.append(data_11)
    support_data_set.append(data_33)
    support_data_set.append(data_55)
    if way == 0:
        Qdata_00=tf.concat([empty_label,Qdata_0],0)
        query=tf.expand_dims(Qdata_00,0)
    elif way == 1:
        Qdata_11=tf.concat([empty_label,Qdata_1],0)
        query=tf.expand_dims(Qdata_11,0)
    elif way == 2:
        Qdata_33=tf.concat([empty_label,Qdata_3],0)
        query=tf.expand_dims(Qdata_33,0)
    else:
        Qdata_55=tf.concat([empty_label,Qdata_5],0)
        query=tf.expand_dims(Qdata_55,0)
    return query

def network_initializer():   
    ini_w1=tf.Variable(tf.random.uniform([788,400],-1,1), trainable=True)      
    ini_w2=tf.Variable(tf.random.uniform([400,200],-1,1), trainable=True)
    ini_w3=tf.Variable(tf.random.uniform([200,10],-1,1), trainable=True)
    return ini_w1, ini_w2, ini_w3

def forward_pass(input,fw1,fw2,fw3):
    h1=tf.matmul(input,fw1)    
    h1=tf.keras.activations.sigmoid(h1)  
    h2=tf.matmul(h1,fw2)    
    h2=tf.keras.activations.sigmoid(h2) 
    out=tf.matmul(h2,fw3)
    out=tf.keras.activations.sigmoid(out)
    return out,fw1,fw2,fw3

def backward_pass(eQD_buffer,bw1,bw2,bw3):
    bw3=tf.transpose(bw3)
    h2=tf.matmul(eQD_buffer,bw3)
    h2=tf.keras.activations.sigmoid(h2)
    bw2=tf.transpose(bw2)
    h1=tf.matmul(h2,bw2)
    h1=tf.keras.activations.sigmoid(h1)
    bw1=tf.transpose(bw1)
    input=tf.matmul(h1,bw1)
    input=tf.keras.activations.sigmoid(input)
    return input

def contrastive_loss(encoded_query_data,cw1,cw2,cw3, similar):
    margin=1.25
    encoded_data,lw1,lw2,lw3 = forward_pass(encoded_query_data,cw1,cw2,cw3)
    retrieved_data, dist_btw_eQD_rD = TCAM_retrieve(encoded_data)
    Dw=Minkowski_distance(encoded_data,retrieved_data)
    loss_value=(similar)*(0.5)*(tf.square(Dw))+(1-similar)*tf.square((0.5)*(tf.math.maximum(0.,margin-Dw)))
    
    return loss_value

def grad(query_input,weight1,weight2,weight3):
    with tf.GradientTape() as tape:
        loss=contrastive_loss(query_input,weight1,weight2,weight3,1)        
    dw1, dw2, dw3 = tape.gradient(loss,[weight1,weight2,weight3])       
    return dw1,dw2,dw3, loss
 
def TCAM_store(encoded_data):
    TCAM_array.append(encoded_data)
    return 0
  
def TCAM_retrieve(encoded_query_data):
    dist1=[]
    dist2=[100.]     
    for stored_data in TCAM_array:        
        dist1.append(Minkowski_distance(encoded_query_data, stored_data))
    min_dist_value=min(dist1)
    min_dist_index=dist1.index(min_dist_value)
    return TCAM_array[min_dist_index], min_dist_value, min_dist_index

def Minkowski_distance(x,y):
    dist=tf.sqrt(tf.reduce_sum(tf.square(x-y)))    
    return dist

query_way=2
query_input=data_preprocessing(query_way)

#sequence
w1, w2, w3 = network_initializer()

#meta_testing
#store support set
for support_input in support_data_set:
    encoder_buffer,w1,w2,w3 = forward_pass(support_input, w1, w2, w3)
    TCAM_store(encoder_buffer)

#query
encoder_buffer,w1,w2,w3 = forward_pass(query_input,w1,w2,w3)
TCAM_buffer, distance_btw_query_mostsimilarTCAM, which_way = TCAM_retrieve(encoder_buffer)
print("retrieved_way: ", which_way)

"""
#backward pass
backward_output=backward_pass(TCAM_buffer,w1,w2,w3)
print("way_0:",way_0)
print("backward_output:", tf.slice(backward_output,[0,0],[1,4]))

#meta_training
for epoch in range(10):       
    grad_weight1,grad_weight2,grad_weight3,loss_value = grad(query_input,w1,w2,w3)
    #print("grad:", grad_weight1)
    if (epoch % 5) == 0:
        print("loss:",loss_value)
    #print("grad sum:", tf.reduce_sum(grad_weight1))
    opt.apply_gradients(zip([grad_weight1,grad_weight2,grad_weight3],[w1,w2,w3]))

#query
encoder_buffer,w1,w2,w3 = forward_pass(query_input,w1,w2,w3)
TCAM_buffer, distance_btw_query_mostsimilarTCAM = TCAM_retrieve(encoder_buffer)
#backward pass
backward_output=backward_pass(TCAM_buffer,w1,w2,w3)

backward_output_label=tf.slice(backward_output,[0,0],[1,4])
backward_output_label=tf.squeeze(backward_output_label)
backward_output_label=tf.argmax(backward_output_label)
print("way_0:",way_0)
print("backward_output:", tf.slice(backward_output,[0,0],[1,4]))
print("label:",backward_output_label)

query_label=tf.argmax(way_0)
print("q_label:", query_label)

#validation

"""