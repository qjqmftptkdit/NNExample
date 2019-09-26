import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

dataMount = 1000
step = 3

def createDataSet() :
    x = np.arange(0, dataMount)
    y = np.sin(x)

    dataX = []
    dataY = []
    for i in range(dataMount-step) :
        dataX.append(y[i:i+step])
        dataY.append(y[i+step])
    dataX = np.array(dataX)
    dataX = dataX.reshape(dataX.shape[0], 1, dataX.shape[1])
    dataY = np.array(dataY)

    return dataX, dataY

def createRNNModel() :
    model = Sequential()

    model.add(SimpleRNN(units=32, input_shape=(1, step), activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model

def trainModel(dataX, dataY, model) :
    model.fit(dataX, dataY, epochs=100, batch_size=16, verbose=2)

def testModel(dataX, dataY, model) :
    score = model.evaluate(dataX, dataY, verbose=0)
    print("Score : {}".format(score))

def predictModel(dataX, dataY, model) :
    result = model.predict(dataX)

    plt.plot(dataY, "-b", result, "-r")
    plt.show()

if __name__ == "__main__":
    dataX, dataY = createDataSet()
    model = createRNNModel()

    trainModel(dataX, dataY, model)
    testModel(dataX, dataY, model)
    predictModel(dataX, dataY, model)


