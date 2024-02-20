# Load MNIST data
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.utils import plot_model
import numpy as np
import time
from datetime import timedelta


def build_network(input_train, filter_num, ker1_size, ker2_size, dense_size):
    inputs = Input(shape=input_train.shape[1:])
    x = Conv2D(filters=filter_num, kernel_size=(ker1_size, ker1_size), activation='relu')(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=filter_num, kernel_size=(ker2_size, ker2_size), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    net = Model(inputs=inputs, outputs=outputs)
    # Compile the network
    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return net


def test_network_kmeans(epoch_num, filter_num, ker1_size, ker2_size, dense_size):
    f = open("output.txt", "a")
    f.write(f'NEW VERSION\n######\n Epochs: {epoch_num}, Filters: {filter_num}, Kernels: {ker1_size},{ker2_size}, Dense: {dense_size}\n')
    (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
    # Preprocess the data from int (0-255) to float (0.0-1.0)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # To reshape data into a convoluted layer
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Model configuration
    batch_size = 50
    img_width, img_height, img_num_channels = 28, 28, 1
    # Determine shape of the data
    input_shape = (img_width, img_height, img_num_channels)

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Merge inputs and targets
    inputs = np.concatenate((x_train, x_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=6, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        # Define the model architecture
        model = build_network(inputs[train], filter_num, ker1_size, ker2_size, dense_size)
        #model = load_model("network_for_mnist.h5")
        # Generate a print
        print('------------------------------------------------------------------------\n')
        print(f'Training for fold {fold_no} ...\n')

        # Fit data to model
        history = model.fit(inputs[train], targets[train],
                            batch_size=batch_size,
                            epochs=epoch_num,
                            verbose=2)

        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        #f.write(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%\n')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    f.write('------------------------------------------------------------------------\n')
    f.write('Score per fold\n')
    for i in range(0, len(acc_per_fold)):
        f.write('------------------------------------------------------------------------\n')
        f.write(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%\n')
    f.write('------------------------------------------------------------------------\n')
    f.write('Average scores for all folds:\n')
    f.write(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})\n')
    f.write(f'> Loss: {np.mean(loss_per_fold)}\n')
    f.write('------------------------------------------------------------------------\n')
    f.close()


#test_network_kmeans(epoch_num=15, filter_num=32, ker1_size=5, ker2_size=3, dense_size=10)
start = time.time()
f_param = open("testparams.txt", "rt")
version = 0
for line in f_param:
    if version == 0:
        print(str(line))
    else:
        params = line.split(';')
        test_network_kmeans(epoch_num=int(params[0]), filter_num=int(params[1]), ker1_size=int(params[2]),
                            ker2_size=int(params[3]), dense_size=int(params[4]))
    version += 1
f_param.close()
end = time.time()
print(f"Elapsed time: {timedelta(seconds=end-start)}")
