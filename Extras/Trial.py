# from tensorflow.keras.models import Model
# from tensorflow.python.keras.layers import Input, GRU
# from tensorflow.python.keras.layers import LSTM
# from numpy import array
# from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# # define model
# # model = Sequential()
# # model.add(LSTM (1, input_shape=(3,4), return_sequences= True))
# # #model.add(Dense( 4,  activation="softmax" ))
# # print(model.summary())
# #
# # # define input data
# # data = array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.7, 0.2, 0.8], [0.54, 0.23, 0.44, 0.38]]).reshape((1,3,4))
# #
# # print(array([0.1, 0.2, 0.3]).reshape(1,3,1))
# # # make and show prediction
# # print("Predict data")
# # prediction = model.predict(data)
# # print(prediction)
# # print(prediction.shape)
# #
# # print("Prediction for all timesteps")
# # model2 = Sequential()
# # model2.add(LSTM (1, input_shape=(12,1), return_sequences= True))
# # data2 = array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.7, 0.2, 0.8], [0.54, 0.23, 0.44, 0.38]]).reshape((1,12,1))
# # print(model2.predict(data2))
#
# #Do predict for each timestep in a 3d array, first split all the inner arrays into smaller arrays
# #of individual numbers then calculate the hidden states for each
#
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.layers import LSTM
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import Input
# from tensorflow.python.keras.models import Model
# from pickle import load
#
# #I am going to change this to a functional api to enable me get the cell states and all of that
#
# arrayShape = open("Text Files and Dictionary/arrayShape.txt", "r")
# shape = arrayShape.read().split("\n")
#
#
# xTrainShape = tuple(list(map(int, shape[0].split(" "))))
# #print(xShape)
# yTrainShape = tuple(list(map(int, shape[1].split(" "))))
# #print(yShape)
# xTestShape = tuple(list(map(int, shape[2].split(" "))))
# #print(xShape)
# yTestShape = tuple(list(map(int, shape[3].split(" "))))
# #print(yShape)
#
#
# x_train = np.fromfile("Text Files and Dictionary/xTrain.bin", dtype=np.float32)
# x_train = x_train.reshape(xTrainShape)
#
# y_train= np.fromfile("Text Files and Dictionary/yTrain.bin", dtype=np.float32)
# y_train = y_train.reshape(yTrainShape)
#
# x_test= np.fromfile("Text Files and Dictionary/xTest.bin", dtype=np.float32)
# x_test = x_test.reshape(xTestShape)
#
#
# y_test= np.fromfile("Text Files and Dictionary/yTest.bin", dtype=np.float32)
# y_test = y_test.reshape(yTestShape)
#
# dictionary = load(open("Text Files and Dictionary/dictionarySP.pkl", "rb"))
#
# # model = Sequential()
# # model.add(LSTM (90, input_shape=(x_train.shape[1], x_train.shape[2])))
# # model.add(Dense (len(dictionary), activation="softmax"))
# # print(model.summary())
# # #How do you choose the amount of memory cells for the LSTM?
# # #How do you know what activation is best
# #
# # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# # history = model.fit(x_train, y_train, epochs= 50, verbose=2, validation_data=(x_test, y_test))
# # results = model.evaluate(x_test, y_test)
# # print("test loss, test acc:", results)
# #
# # input1 = Input(shape= (x_train.shape[1], x_train.shape[2]))
# # lstm2, state_h1, state_c1 = LSTM(90, return_state=True)(input1)
# # dense2 = Dense(len(dictionary), activation="softmax")(lstm2)
# # model4 = Model(inputs=input1, outputs=dense2)
# # print(model4.summary())
# #
# #
# # input = Input(shape= (x_train.shape[1], x_train.shape[2]))
# # lstm1, state_h, state_c = LSTM(90, return_state=True)(input)
# # dense = Dense(len(dictionary), activation="softmax")(lstm1)
# # model2 = Model(inputs=input, outputs=[dense, state_h, state_c])
# # print(model2.summary())
#
# #
# # model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# # #history = model2.fit(x_train, y_train, epochs= 100, verbose=2, validation_data=(x_test, y_test))
# # # results = model2.evaluate(x_test, y_test)
# # # print("test loss, test acc:", results)
# #
# # model2.save("Learned Models/learnedLSTMFunc.h5")
# #
#
# # modelGRU = Sequential()
# # modelGRU.add(GRU (90, input_shape=(x_train.shape[1], x_train.shape[2])))
# # modelGRU.add(Dense (len(dictionary), activation="softmax"))
# # print(modelGRU.summary())
# #
# #
# # input2 = Input(shape= (x_train.shape[1], x_train.shape[2]))
# # gru1 = GRU(90)(input2)
# # dense2 = Dense(len(dictionary), activation="softmax")(gru1)
# # model3 = Model(inputs=input2, outputs=dense2)
# # print(model3.summary())
#
#
# # model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# # history = model3.fit(x_train, y_train, epochs= 100, verbose=2, validation_data=(x_test, y_test))
# # results3 = model3.evaluate(x_test, y_test)
# # print("test loss, test acc:", results3)
#
# # modelGRU.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# # history = modelGRU.fit(x_train, y_train, epochs= 100, verbose=2, validation_data=(x_test, y_test))
# # resultsGRU = modelGRU.evaluate(x_test, y_test)
# # print("test loss, test acc:", resultsGRU)


import csv

rowData = {
    "SequenceLength" : 5,
    "LSTM train accuracy take 1": 0,
    "LSTM train accuracy take 2": 0,
    "LSTM train accuracy average": 0,
    "GRU train accuracy take 1": 0,
    "GRU train accuracy take 2": 0,
    "GRU train accuracy average": 0,
    "LSTM test accuracy take 1": 0,
    "LSTM test accuracy take 2": 0,
    "LSTM test accuracy average": 0,

}
fieldNames = []
for x in rowData:
    fieldNames.append(x)

file =  open("../Trial.csv", "w", newline="")
writer = csv.DictWriter(file, fieldnames=fieldNames)
writer.writeheader()
writer.writerow(rowData)

for x in [1, 3, 5,7, 8]:
    rowData["SequenceLength"] = x
    writer.writerow(rowData)

file.close()