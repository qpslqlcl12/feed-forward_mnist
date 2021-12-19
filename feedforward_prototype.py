import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import time

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
batch_size=32
#print("train_label = ", y_train[1])
#x_train_one_image = x_train[0,]
#print(x_train_one_image)

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))      #[x_train, y_train]
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)   #put in buffer 1024 data
#print(train_dataset)

finputs = keras.Input(shape=(784,))
fx1 = layers.Dense(400, activation="sigmoid")(finputs)
fx2 = layers.Dense(200, activation="sigmoid")(fx1)
encoder_outputs = layers.Dense(10, activation="sigmoid")(fx2)
recognition_model = keras.Model(inputs=finputs, outputs=encoder_outputs, name="recognition_model")

binputs = keras.Input(shape=(784,))
bx1 = layers.Dense(400, activation="sigmoid")(binputs)
bx2 = layers.Dense(200, activation="sigmoid")(bx1)
decoder_outputs = layers.Dense(10, activation="sigmoid")(bx2)
retrieve_model = keras.Model(inputs=binputs, outputs=decoder_outputs, name="retrieve_model")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
""""
def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)        # logit(y)= log(y/(1-y))
    
    return loss_object(y_true=y, y_pred=y_)

l = loss(model, x_train, y_train, training=False)
print("Loss test: {}".format(l))
""""
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        #loss_value = loss(model, inputs, targets, training=True)
        loss_value = read_phase(inputs)
    return loss_value, tape.gradient(loss_value, model.trainable_variables) #dloss_value/dmodel.trainable_variables

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_value, grads = grad(model, x_train, y_train)
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, x_train, y_train, training=True).numpy()))


## Note: Rerunning this cell uses the same model variables



def read_phase(inputs):
    encoded_data = recognition_model(inputs, training = False)
    #encoded_data goes in to TCAM
    #retrieved data from TCAM is input of retrieved_model and source of loss function.
    retrieved_data = retrieve_model(encoded_data, training = False)
    #label neuron of input layer need for loss calculation 
    return loss_object(y_true=y, y_pred=retrieved_data)

def write_phase():
    loss_value, grads = grad(retrieve_model, TCAM_out_data, true_label) # needs label side of encoded_data
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #weight transfer
    #see keras.layers.Layer.get_weights()
    #TCAM write phase

""""


# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 1

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    
  # Training loop - using batches of 32
    for x, y in train_dataset:

      # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, model(x, training=True))

        #if epoch % 1 == 0:
            #print("label =", y)
    
    loss_value, grads = grad(model, x_train, y_train)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg.update_state(loss_value)  
    epoch_accuracy.update_state(y_train, model(x_train, training=True))
    #print("label = ", y_train)
    #print(len(y_train))
    #print("weight:",model.trainable_variables)

  # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

""""                                                           