# Load MNIST data
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
# Preprocess the data from int (0-255) to float (0.0-1.0)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)
# To reshape data into a dense layer
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# To reshape data into a convoluted layer
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create a net model
inputs = Input(shape=x_train.shape[1:])
x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(rate=0.5)(x)
outputs = Dense(10, activation='softmax')(x)
net = Model(inputs=inputs, outputs=outputs)

# net.summary()
# plot_model(net, to_file='network_structure.png', show_shapes=True)
# Compile the network
net.compile(loss='categorical_crossentropy', optimizer='adam')
history = net.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs=10,
                  batch_size=256)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# Save
net.save("network_for_mnist.h5")

# Load
#from keras.models import load_model
#net=load_model("network_for_mnist.h5")

# Test
outputs=net.predict(x_test)
labels_predicted=np.argmax(outputs, axis=1)
misclassified=sum(labels_predicted!=labels_test)
print('Percentage misclassified = ',100*misclassified/labels_test.size)

