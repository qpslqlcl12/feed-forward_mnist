import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,))
dense = layers.Dense(784, activation="sigmoid")
x = dense(inputs)
x = layers.Dense(400, activation="sigmoid")(x)
outputs = layers.Dense(10)(x)


model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
#model.summary()


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()




"""
plt.figure()
plt.imshow(x_train[0])
plt.savefig('test.png', dpi=500)
print('train label=', y_train[0])
"""

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255



model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)


history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=0)

print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc.png')
"""



#model.fit(train_images, train_labels, epochs=1)  
#model.evaluate(train_images, train_labels)
#model.train_on_batch(train_images, train_labels )
#model.test_on_batch(train_images, train_labels)
