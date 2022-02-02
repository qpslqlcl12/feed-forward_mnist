import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os

start_time = time.time()

np.set_printoptions(precision=6, suppress=True)
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

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

#label for 4 way 
way_0 = tf.Variable([1.,0.,0.,0.])
way_1 = tf.Variable([0.,1.,0.,0.])
way_2 = tf.Variable([0.,0.,1.,0.])
way_3 = tf.Variable([0.,0.,0.,1.])
way.append(way_0)
way.append(way_1)
way.append(way_2)
way.append(way_3)


def load_img(fn):
    I=plt.imread(fn)
    I=np.array(I,dtype=bool)
    I=np.invert(I)
    I=I.astype('float32')
    I=I.flatten()
    #print(I.shape)
    return I


def testing_data_set_sampling():
    testing_img_dir = '../omniglot/python/images_evaluation'
    nalpha = 4 # number of alphabets to show
    alphabet_names = [a for a in os.listdir(testing_img_dir) if a[0] != '.'] # get folder names
    alphabet_names = random.sample(alphabet_names,nalpha) # choose random alphabets
    #print("alpha")
    #print(alphabet_names)
    pick_query=random.randint(0,3)
   
    sampling_support_data_set=[]
    #sampling_query_data_set=[]
    random_label_list=alphabet_names 
    #print(random_label_list)
    for character in alphabet_names:
        character_id = random.randint(1,len(os.listdir(os.path.join(testing_img_dir,character))))
        string=str(character_id)
        img_char_dir = os.path.join(testing_img_dir,character,'character'+ string.zfill(2))
        support=random.randint(0,15)        
       
        support_set=os.listdir(img_char_dir)[support]           
       
        support_image=img_char_dir + '/' + support_set        
        #print(support_image)
        testing_support_set=load_img(support_image)     
        
        sampling_support_data_set.append(testing_support_set)        
        
        if alphabet_names[pick_query] == character:
            query=random.randint(16,19)
            query_set=os.listdir(img_char_dir)[query]   
            query_image=img_char_dir + '/' + query_set
            query_label=character
    testing_query_set=load_img(query_image)
    #print("test query set")
    #print(query_image)  
    sampling_query_way=random_label_list.index(query_label)
    #print("check")
    #print(sampling_query_way)
        

    return random_label_list,sampling_support_data_set,query_label,testing_query_set,sampling_query_way

def training_data_set_sampling():
    training_img_dir = '../omniglot/python/images_background'    
    nalpha = 4 # number of alphabets to show
    alphabet_names = [a for a in os.listdir(training_img_dir) if a[0] != '.'] # get folder names
    alphabet_names = random.sample(alphabet_names,nalpha) # choose random alphabets
    #print(alphabet_names) # 4 kinds of random alphabet
    sampling_support_data_set=[]
    sampling_query_data_set=[]
    random_label_list=alphabet_names 
    #print(random_label_list)
    for character in alphabet_names:
        character_id = random.randint(1,len(os.listdir(os.path.join(training_img_dir,character))))
        string=str(character_id)
        img_char_dir = os.path.join(training_img_dir,character,'character'+ string.zfill(2))
        support=random.randint(0,15)
        query=random.randint(16,19)
       
        support_set=os.listdir(img_char_dir)[support]
        query_set=os.listdir(img_char_dir)[query]       
       
        support_image=img_char_dir + '/' + support_set
        query_image=img_char_dir + '/' + query_set
        
        training_support_set=load_img(support_image)
        training_query_set=load_img(support_image)
        
        sampling_support_data_set.append(training_support_set)
        sampling_query_data_set.append(training_query_set)

    #print(sampling_support_data_set)
    #print(sampling_query_data_set)  
    sampling_query_way=[0,1,2,3]        
   
    return random_label_list,sampling_support_data_set,query_set,sampling_query_data_set, sampling_query_way



def data_preprocessing_for_testing(way_buf, data_set, q_data_set):
    empty_label = tf.Variable([0.,0.,0.,0.])
    data_buffer=[]    
    for i,j in zip(way_buf,data_set):
        #print("way_buf",way_buf)
        #print("data_set",data_set)
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
    ini_w1=tf.Variable(tf.random.uniform([11029,5000],-1,1), trainable=True)      
    ini_w2=tf.Variable(tf.random.uniform([5000,1500],-1,1), trainable=True)
    ini_w3=tf.Variable(tf.random.uniform([1500,400],-1,1), trainable=True)
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
    margin=100.0
    encoded_data,lw1,lw2,lw3 = forward_pass(encoded_query_data,cw1,cw2,cw3)
    retrieved_data, dist_btw_eQD_rD, retrieved_way, dist_btw_Q_TCAM = TCAM_retrieve(encoded_data)
    Dw=Euclidian_distance(encoded_data,retrieved_data)
    
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
        dist1.append(Euclidian_distance(encoded_query_data, stored_data))
    min_dist_value=min(dist1)
    min_dist_index=dist1.index(min_dist_value)
    return TCAM_array[min_dist_index], min_dist_value, min_dist_index, dist1

def Euclidian_distance(x,y):
    dist=tf.sqrt(tf.reduce_sum(tf.square(x-y)))    
    return dist



w1, w2, w3 = network_initializer()


#sequence
def meta_testing(weight_buf1,weight_buf2,weight_buf3, save_trigger):
#meta_testing = 100 
    count=0
    answer=0
    repeat=0
    for testing in range(100):
        count=count+1
        support_label_list,support_buff,query_label,query_data_buff,query_way = testing_data_set_sampling()
        support_data_set,query_data=data_preprocessing_for_testing(way,support_buff,query_data_buff) 
    #store support set
        for support_input,label in zip(support_data_set,support_label_list):
            encoder_buffer,w1,w2,w3 = forward_pass(support_input, weight_buf1,weight_buf2,weight_buf3)
            TCAM_store(encoder_buffer)

            if (save_trigger ==1) and testing == 99 :
                qd=tf.slice(support_input,[0,3],[1,11025])
                qd=tf.reshape(qd,[105,105])   
                plt.imshow(qd, cmap='gray')
                repeat=str(label)
                plt.savefig('./figures/'+repeat+'_support_input_init.png')

                edb=tf.reshape(encoder_buffer,[20,20])
                plt.imshow(edb, cmap='gray')
                plt.savefig('./figures/'+repeat+'_encoder_buffer_init.png')                

            if save_trigger == 2 and testing == 99:
                qd=tf.slice(support_input,[0,3],[1,11025])
                qd=tf.reshape(qd,[105,105])   
                plt.imshow(qd, cmap='gray')
                repeat=str(label)
                plt.savefig('./figures/'+repeat+'_support_input_end.png')

                edb=tf.reshape(encoder_buffer,[20,20])
                plt.imshow(edb, cmap='gray')
                plt.savefig('./figures/'+repeat+'_encoder_buffer_end.png')



    #query
        encoder_buffer,w1,w2,w3 = forward_pass(query_data,w1,w2,w3)
        TCAM_buffer, distance_btw_query_mostsimilarTCAM, which_way, dist_btw_Q_TCAM = TCAM_retrieve(encoder_buffer)
        if(testing%50 == 0):
            print("testing_support_label",support_label_list)
            print("query_label", query_label)
            print("query_way",query_way)
            print("retrieved_way: ", which_way)  
            print("distances btw query & TCAM stored_data", dist_btw_Q_TCAM)
        if query_way == which_way:
            answer=answer+1
        accuracy=answer/count

        TCAM_array.clear()
    return accuracy

def meta_training(weight_buf1,weight_buf2,weight_buf3):
    
#meta_training, in training seq, query is set of all the ways.
    support_label_list,support_buff,query_label,query_data_buff,query_way = training_data_set_sampling() 
    support_data_set,query_data=data_preprocessing_for_training(way,support_buff,query_data_buff) 
    #print("support_set")
    #print(support_label_list)
    #print(support_data_set)
    #print(query_label)

     
    for support_input in support_data_set:    
        encoder_buffer,weight_buf1,weight_buf2,weight_buf3 = forward_pass(support_input,weight_buf1,weight_buf2,weight_buf3)
        TCAM_store(encoder_buffer)

    for query_data_sample,query_way_list in zip(query_data,query_way):
        grad_weight1,grad_weight2,grad_weight3,loss_value = grad(query_data_sample,weight_buf1,weight_buf2,weight_buf3,query_way_list)
        print("loss:",loss_value)                
        opt.apply_gradients(zip([grad_weight1,grad_weight2,grad_weight3],[weight_buf1,weight_buf2,weight_buf3]))
        
    TCAM_array.clear()
    
    return weight_buf1,weight_buf2,weight_buf3, loss_value


#testing_data_set_sampling()


#ac=meta_testing(w1,w2,w3)
#print(ac)
#w1,w2,w3=meta_training(w1,w2,w3)
accuracy_list=[]
loss_list=[]
epoch=10000
max_acc=0
min_loss=0
trigger=0
for epoches in range(epoch):
    print('=======',epoches,'th epoch========')
    if epoches == 1:
        trigger=1
    if epoches == (epoch-2):
        trigger=2
    acc=meta_testing(w1,w2,w3,trigger)
    trigger=0
    w1,w2,w3,loss_buf=meta_training(w1,w2,w3)
    loss_list.append(loss_buf)
    print("accuracy:",acc)
    accuracy_list.append(acc)
    if max_acc < acc:
        max_acc = acc
    if min_loss > loss_buf:
        min_loss = loss_buf
epoch=range(0,epoch)
plt.subplot(2,1,1)
plt.plot(epoch, accuracy_list)
plt.title('accuracy')
plt.subplot(2,1,2)
plt.plot(epoch, loss_list)
plt.title('loss')


plt.savefig('test.png', dpi=500)

print("maximum_accuracy", max_acc)
print("minmum_loss", loss_buf)

print("Running time: {:.4f}min".format((time.time()-start_time)/60))
#end of code


