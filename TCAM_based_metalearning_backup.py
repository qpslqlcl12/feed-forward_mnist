import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import random
import matplotlib.pyplot as plt

start_time = time.time()

np.set_printoptions(precision=6, suppress=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
"""
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
"""

#label for 4 way 
way_0 = tf.Variable([1.,0.,0.,0.])
way_1 = tf.Variable([0.,1.,0.,0.])
way_2 = tf.Variable([0.,0.,1.,0.])
way_3 = tf.Variable([0.,0.,0.,1.])


way=[]
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
opt=tf.keras.optimizers.Adam(learning_rate=0.001)

way.append(way_0)
way.append(way_1)
way.append(way_2)
way.append(way_3)


def testing_data_set_sampling():
    random_label_list=set(y_test)
    random_label_list=random.sample(random_label_list,4)
    #print("training_label_list:", random_label_list)
    sampling_support_data_set=[]
    for i in random_label_list: 
        while len(sampling_support_data_set) <= 4:
            random_label=random.randint(0,9999)
            if i == y_test[random_label]:            
                sampling_support_data_set.append(x_test[random_label])           
                break
    query_label=random.sample(random_label_list,1)
    #print("q_label",query_label)
    sampling_query_way=random_label_list.index(query_label)
    #print("queryway", query_way)
    query_data_set=[]
    for i in query_label: 
        while len(query_data_set) <= 1:
            random_label=random.randint(0,9999)
            if i == y_test[random_label]:            
                query_data_set.append(x_test[random_label])           
                break
    #print("query data set",query_data_set)

    return random_label_list,sampling_support_data_set,query_label,query_data_set,sampling_query_way

def training_data_set_sampling():
    random_label_list=set(y_train)
    random_label_list=random.sample(random_label_list,4)
    #print("training_label_list:", random_label_list)
    sampling_support_data_set=[]
    for i in random_label_list: 
        while len(sampling_support_data_set) <= 4:
            random_label=random.randint(0,59999)
            if i == y_train[random_label]:            
                sampling_support_data_set.append(x_train[random_label])           
                break

    query_label=random_label_list
    #print("q_label",query_label)
    sampling_query_way=[0,1,2,3]
    #print("queryway", query_way)
    query_data_set=[]
    for i in query_label: 
        while len(query_data_set) <= 4:
            random_label=random.randint(0,59999)
            if i == y_train[random_label]:            
                query_data_set.append(x_train[random_label])           
                break
    #print("query data set",query_data_set)

    return random_label_list,sampling_support_data_set,query_label,query_data_set, sampling_query_way



def data_preprocessing_for_testing(way_buf, data_set, q_data_set):
    empty_label = tf.Variable([0.,0.,0.,0.])
    data_buffer=[]    
    for i,j in zip(way_buf,data_set):
        sampled_data=tf.concat([i,j],0)
        sampled_data=tf.expand_dims(sampled_data,0)
        data_buffer.append(sampled_data)
    sampled_q_data_set=tf.concat([empty_label,tf.squeeze(q_data_set)],0)    
    sampled_q_data_set=tf.expand_dims(sampled_q_data_set,0)
    return data_buffer,sampled_q_data_set

def data_preprocessing_for_training(way_buf, data_set, q_data_set):
    empty_label = tf.Variable([0.,0.,0.,0.])
    data_buffer=[]    
    for i,j in zip(way_buf,data_set):
        sampled_data=tf.concat([i,j],0)
        sampled_data=tf.expand_dims(sampled_data,0)
        data_buffer.append(sampled_data)

    query_data_buffer=[]
    for l in q_data_set:
        sampled_query_data=tf.concat([empty_label,l],0)
        sampled_query_data=tf.expand_dims(sampled_query_data,0)
        query_data_buffer.append(sampled_query_data)
    return data_buffer,query_data_buffer

def network_initializer():   
    ini_w1=tf.Variable(tf.random.uniform([788,500],-1,1), trainable=True)      
    ini_w2=tf.Variable(tf.random.uniform([500,300],-1,1), trainable=True)
    ini_w3=tf.Variable(tf.random.uniform([300,100],-1,1), trainable=True)
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

def contrastive_loss(encoded_query_data,cw1,cw2,cw3,query_way_buff):
    margin=1.0
    encoded_data,lw1,lw2,lw3 = forward_pass(encoded_query_data,cw1,cw2,cw3)
    retrieved_data, dist_btw_eQD_rD, retrieved_way = TCAM_retrieve(encoded_data)
    Dw=Minkowski_distance(encoded_data,retrieved_data)
    
    if query_way_buff==retrieved_way:
        similar = 1
        print("similar")
    else:
        similar = 0
        print("dissimilar")
    
    #similar=0
    loss_value=(similar)*(0.5)*(tf.square(Dw))+(1-similar)*tf.square((0.5)*(tf.math.maximum(0.,margin-Dw)))
    
    return loss_value

def grad(query_input,weight1,weight2,weight3,query_way_buff):
    with tf.GradientTape() as tape:
        loss=contrastive_loss(query_input,weight1,weight2,weight3,query_way_buff)        
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



w1, w2, w3 = network_initializer()


#sequence
def meta_testing(weight_buf1,weight_buf2,weight_buf3):
#meta_testing = 100 
    count=0
    answer=0
    for testing in range(100):
        count=count+1
        support_label_list,support_buff,query_label,query_data_buff,query_way = testing_data_set_sampling()
        support_data_set,query_data=data_preprocessing_for_testing(way,support_buff,query_data_buff)  

        

    #store support set
        for support_input in support_data_set:
            encoder_buffer,w1,w2,w3 = forward_pass(support_input, weight_buf1,weight_buf2,weight_buf3)
            TCAM_store(encoder_buffer)

    #query
        encoder_buffer,w1,w2,w3 = forward_pass(query_data,w1,w2,w3)
        TCAM_buffer, distance_btw_query_mostsimilarTCAM, which_way = TCAM_retrieve(encoder_buffer)
        if(testing%50 == 0):
            print("support_label",support_label_list)
            print("query_way",query_way)
            print("retrieved_way: ", which_way)  
        if query_way == which_way:
            answer=answer+1
        accuracy=answer/count
        #print("accuracy:",accuracy )

        TCAM_array.clear()
    return accuracy

def meta_training(weight_buf1,weight_buf2,weight_buf3):
    count=0
#meta_training, in training seq, query is set of all the ways.
    support_label_list,support_buff,query_label,query_data_buff,query_way = training_data_set_sampling() 
    support_data_set,query_data=data_preprocessing_for_training(way,support_buff,query_data_buff) 

    for support_input in support_data_set:    
        encoder_buffer,weight_buf1,weight_buf2,weight_buf3 = forward_pass(support_input,weight_buf1,weight_buf2,weight_buf3)
        TCAM_store(encoder_buffer)

    for query_data_sample,query_way_list in zip(query_data,query_way):
        grad_weight1,grad_weight2,grad_weight3,loss_value = grad(query_data_sample,weight_buf1,weight_buf2,weight_buf3,query_way_list)
        print("loss:",loss_value)                
        opt.apply_gradients(zip([grad_weight1,grad_weight2,grad_weight3],[weight_buf1,weight_buf2,weight_buf3]))
        count=count+1
    TCAM_array.clear()
    #print("count",count)
    return weight_buf1,weight_buf2,weight_buf3, loss_value

#ac=meta_testing(w1,w2,w3)
#print(ac)
#w1,w2,w3=meta_training(w1,w2,w3)
accuracy_list=[]
loss_list=[]
epoch=10000
max_acc=0
for epoches in range(epoch):
    acc=meta_testing(w1,w2,w3)
    w1,w2,w3,loss_buf=meta_training(w1,w2,w3)
    #if epoches % 10 == 0:
    loss_list.append(loss_buf)
    print("accuracy:",acc)
    accuracy_list.append(acc)
    if max_acc < acc:
        max_acc = acc
epoch=range(0,epoch)
plt.subplot(2,1,1)
plt.plot(epoch, accuracy_list)
plt.title('accuracy')
plt.subplot(2,1,2)
plt.plot(epoch, loss_list)
plt.title('loss')


plt.savefig('test.png', dpi=500)

print("maximum_accuracy", max_acc)


print("Running time: {:.4f}min".format((time.time()-start_time)/60))
#end of code
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