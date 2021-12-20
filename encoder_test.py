import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import time
import pandas as pd


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#print(x_train.shape)
#df = pd.DataFrame(x_train[0])
#df.to_excel('test.xlsx', index=False)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
print("x_train shape", x_train.shape)
x_test = x_test.reshape(10000, 784).astype("float32") / 255
batch_size=32
#print("train_label = ", y_train[1])
#x_train_one_image = x_train[0, -1]
#x_train_one_image = tf.reshape(x_train_one_image, [-1])
#print(x_train_one_image.shape)

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))      #[x_train, y_train]
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)   #put in buffer 1024 data
#print(train_dataset)

finputs = keras.Input(shape=(784,))
fx1 = layers.Dense(64, activation="sigmoid")(finputs)
fx2 = layers.Dense(64, activation="sigmoid")(fx1)
encoder_outputs = layers.Dense(10)(fx2)
model = keras.Model(inputs=finputs, outputs=encoder_outputs, name="recognition_model")


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)        # logit(y)= log(y/(1-y))
    
    return loss_object(y_true=y, y_pred=y_)

l = loss(model, x_train, y_train, training=False)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)        
    return loss_value, tape.gradient(loss_value, model.trainable_variables) #dloss_value/dmodel.trainable_variables

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, x_train, y_train)
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, x_train, y_train, training=True).numpy()))

def one_image():
#    loss_value, grads = grad(model, x_train, y)
#    optimizer.apply_gradients(zip(grads, model.trainable_variables))
#    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
#    epoch_accuracy.update_state(y, model(x, training=True))
    print("test logit : \n")
    print(model.predict(x_train[0, 783]))

one_image()


"""
## Note: Rerunning this cell uses the same model variables

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

  # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

"""