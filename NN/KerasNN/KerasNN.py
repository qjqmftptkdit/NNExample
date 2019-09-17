
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.utils import normalize, to_categorical

# Load data from csv file
def getDataFromCSV() :
    data = pd.read_csv("KerasNN.csv", sep="\t")

    X_train = np.matrix(data["Num"].values.tolist()[:50]).reshape(50, 1)
    X_test = np.matrix(data["Num"].values.tolist()[50:]).reshape(30, 1)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 100
    X_test /= 100

    y_data = data["Class"].values.tolist()
    y_dic = {'A':0, 'B':1, 'C':2}
    for i in range(len(y_data)) :
        y_data[i] = y_dic[y_data[i]]

    y_train = y_data[:50]
    y_test = y_data[50:]
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    return X_train, Y_train, X_test, Y_test

# Create Simple NN model for classifying
def createNNModel() :
    model = Sequential()

    for i in range(5) :
        model.add(Dense(100))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def trainModel(model, X_train, Y_train) :
    model.fit(X_train, Y_train, epochs=1000, batch_size=25, verbose=1, shuffle=True)

def testModel(model, X_test, Y_test) :
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    return score, acc

# Hand test 
def predictModel(model) :
    while True :
        x = int(input("Input Num : "))
        x = np.matrix(x).reshape(1,1)
        x = x.astype("float32")
        x /= 100

        result = model.predict(x)
        result = result.reshape(3)
        maxIndex = np.argmax(result)

        result_dic = {0:'A', 1:'B', 2:'C'}
        print("Predicted Class : {}".format(result_dic[maxIndex]))

if __name__ == "__main__" :
    X_train, Y_train, X_test, Y_test = getDataFromCSV()
    model = createNNModel()
    trainModel(model, X_train, Y_train)
    score, acc = testModel(model, X_test, Y_test)
    print("Score : {}".format(score))
    print("Accuracy : {}".format(acc))

    predictModel(model)
    
