from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np

# Load data
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
# Preprocess the data from int (0-255) to float (0.0-1.0)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

net = load_model("network_for_mnist.h5")

outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
misclassified = sum(labels_predicted != labels_test)
print('Percentage misclassified = ', 100*misclassified/labels_test.size, '%')

plt.figure(figsize=(8, 2))
for i in range(0, 8):
    ax = plt.subplot(2, 8, i+1)
    plt.imshow(x_test[i, :].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    plt.title(labels_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
for i in range(0,8):
    #output = net.predict(x_test[i, :].reshape(1, 784)) #if MLP
    output = net.predict(x_test[i, :].reshape(1, 28, 28, 1)) #if CNN
    output = output[0, 0:]
    plt.subplot(2, 8, 8+i+1)
    plt.bar(np.arange(10.), output)
    plt.title(str(np.argmax(output)))
