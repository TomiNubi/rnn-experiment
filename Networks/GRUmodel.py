import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.layers import Dense
from pickle import load


arrayShape = open("../Text Files and Dictionary/arrayShape.txt", "r")
shape = arrayShape.read().split("\n")


xTrainShape = tuple(list(map(int, shape[0].split(" "))))
#print(xShape)
yTrainShape = tuple(list(map(int, shape[1].split(" "))))
#print(yShape)
xTestShape = tuple(list(map(int, shape[2].split(" "))))
#print(xShape)
yTestShape = tuple(list(map(int, shape[3].split(" "))))
#print(yShape)


x_train = np.fromfile("../Text Files and Dictionary/xTrain.bin", dtype=np.float32)
x_train = x_train.reshape(xTrainShape)

y_train= np.fromfile("../Text Files and Dictionary/yTrain.bin", dtype=np.float32)
y_train = y_train.reshape(yTrainShape)

x_test= np.fromfile("../Text Files and Dictionary/xTest.bin", dtype=np.float32)
x_test = x_test.reshape(xTestShape)


y_test= np.fromfile("../Text Files and Dictionary/yTest.bin", dtype=np.float32)
y_test = y_test.reshape(yTestShape)

dictionary = load(open("../Text Files and Dictionary/dictionarySP.pkl", "rb"))

model = Sequential()
model.add(GRU (90, input_shape=(x_train.shape[1], x_train.shape[2]), reset_after=False))
model.add(Dense (len(dictionary), activation="softmax"))
print(model.summary())
#How do you choose the amount of memory cells for the LSTM?
#How do you know what activation is best

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs= 100, verbose=2, validation_data=(x_test, y_test))
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)

#print(history.history)
model.save("../Learned Models/learnedModelGRU.h5")

