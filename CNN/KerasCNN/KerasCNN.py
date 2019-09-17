import numpy as np
np.random.seed(123) # Fix pseudor random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

# Load Mnist Data and Convert data for adjusting about CNN
def loadDataAndConvert() :
    # Load mnist Data for testing
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape data to get deepth 
    # (n, width, height) to (n, depth, width, height)
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    # Nomarize data 
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    # Convert result to One-Hotpix
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test

# Return CNN train model that is maked by Sequential architecture
def makeTrainModel() :
    # Load proved sequential architecture
    model = Sequential() 
    model.add(Convolution2D(32, (3, 3), activation="relu", input_shape=(1, 28, 28), data_format='channels_first'))
    model.add(Convolution2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Fit model by acquired train data
def fitModel(model, X_train_, Y_train) :
    model.fit(X_train_, Y_train, batch_size=32, nb_epoch=10, verbose=1)

def testModel(model, X_test, Y_test) :
    score = model.evaluate(X_test, Y_test, verbose=0)
    return score


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = loadDataAndConvert()
    model = makeTrainModel()
    fitModel(model, X_train, Y_train)
    score = testModel(model, X_test, Y_test)
    print("Score : {}".format(score))