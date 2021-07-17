import numpy as np
import tensorflow as tf
from numpy import array
from pickle import dump
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, GRU
from tensorflow.python.keras.layers import Dense
from pickle import load
import csv

#first step is to load the data

def load_data(data):
    file = open(data, "r")
    text = file.read()
    file.close()
    return text
def save_data(data, filename):
    file = open(filename, "w")
    file.write(data)
    file.close


def create_sequence(loadFile, seqLen, saveFile):

    text = load_data(loadFile)
    textLines = text.split()
    joinedText = " ".join(textLines)

    #Create the sequence of data that is going to be feed into the RNN. These would later be
    #seperated into input and output vectors
    #What are the three dimensional inputs like?



    sequence = list()
    for i in range(seqLen, len(joinedText)):
        seq = joinedText[i - seqLen : i+1]
        sequence.append(seq)

    data_to_save = "\n".join(sequence)
    save_data(data_to_save, saveFile)

def load_sequence(file):
    sequenceData = load_data(file)
    sequenceList = sequenceData.split("\n")
    return sequenceData, sequenceList

def int_encode(sequenceList):
    integer_sequence = list()
    for line in sequenceList:
        seq = [dictionary[char] for char in line]
        integer_sequence.append(seq)

    #next we need to one hot encode each sequence such that it is a 0-1 vector with length of the vocabulary size
    #with a one representing the point where the character exists.
    integer_sequence = array(integer_sequence)
    x, y = integer_sequence[:, :-1], integer_sequence[:, -1]
    #print(x)
    int_sequences = [to_categorical(num, num_classes=len(dictionary)) for num in x]
    x = array(int_sequences)
    y= to_categorical(y, num_classes=len(dictionary))
    return x, y

arrayShape = open("Text Files and Dictionary/arrayShape.txt", "r")
shape = arrayShape.read().split("\n")


xTrainShape = tuple(list(map(int, shape[0].split(" "))))
#print(xShape)
yTrainShape = tuple(list(map(int, shape[1].split(" "))))
#print(yShape)
xTestShape = tuple(list(map(int, shape[2].split(" "))))
#print(xShape)
yTestShape = tuple(list(map(int, shape[3].split(" "))))
#print(yShape)


x_train = np.fromfile("Text Files and Dictionary/xTrain.bin", dtype=np.float32)
x_train = x_train.reshape(xTrainShape)

y_train= np.fromfile("Text Files and Dictionary/yTrain.bin", dtype=np.float32)
y_train = y_train.reshape(yTrainShape)

x_test= np.fromfile("Text Files and Dictionary/xTest.bin", dtype=np.float32)
x_test = x_test.reshape(xTestShape)


y_test= np.fromfile("Text Files and Dictionary/yTest.bin", dtype=np.float32)
y_test = y_test.reshape(yTestShape)

dictionary = load(open("Text Files and Dictionary/dictionarySP.pkl", "rb"))


maxEpoch = 100
hiddenUnits = [1, 5, 10, 20, 40, 80, 120, 200, 350]
# hiddenUnits = [1, 5, 10]
#sequenceLength = 10
rowData = {
    "Hidden Units" : 1,
    "LSTM train accuracy take 1": 0,
    "LSTM train accuracy take 2": 0,
    "LSTM train accuracy average": 0,
    "GRU train accuracy take 1": 0,
    "GRU train accuracy take 2": 0,
    "GRU train accuracy average": 0,
    "LSTM test accuracy take 1": 0,
    "LSTM test accuracy take 2": 0,
    "LSTM test accuracy average": 0,
    "GRU test accuracy take 1": 0,
    "GRU test accuracy take 2": 0,
    "GRU test accuracy average": 0,
    "LSTM train loss take 1": 0,
    "LSTM train loss take 2": 0,
    "LSTM train loss average": 0,
    "GRU train loss take 1": 0,
    "GRU train loss take 2": 0,
    "GRU train loss average": 0,
    "LSTM test loss take 1": 0,
    "LSTM test loss take 2": 0,
    "LSTM test loss average": 0,
    "GRU test loss take 1": 0,
    "GRU test loss take 2": 0,
    "GRU test loss average": 0,

}
fieldNames = []
for x in rowData:
    fieldNames.append(x)

file = open("Experiment.csv", "w", newline="")
writer = csv.DictWriter(file, fieldnames=fieldNames)
writer.writeheader()



for hiddenUnit in hiddenUnits:
    #toggle the comments to change the current text file

    # create_sequence("Text Files and Dictionary/Sixpence.txt", sequenceLength, "Text Files and Dictionary/Sixpence sequence.txt")
    # create_sequence("Text Files and Dictionary/testData.txt", sequenceLength, "Text Files and Dictionary/testData sequence.txt")
    print("Next Sequence")

    #CHOOSE THE CURRENT TEXT BEING TESTED


    #Now that the file has been sorted, time to introduce the models

    #LSTM
    modelLSTM1 = Sequential()
    modelLSTM1.add(LSTM(hiddenUnit, input_shape=(x_train.shape[1], x_train.shape[2])))
    modelLSTM1.add(Dense(len(dictionary), activation="softmax"))
    print(modelLSTM1.summary())

    #GRU
    modelGRU1 = Sequential()
    modelGRU1.add(GRU(hiddenUnit, input_shape=(x_train.shape[1], x_train.shape[2]), reset_after=False))
    modelGRU1.add(Dense(len(dictionary), activation="softmax"))
    print(modelGRU1.summary())


    #time to populate the dictionary
    rowData["Hidden Units"] = hiddenUnit

    #FOR THE TAKE 1s

    #compile LSTM
    modelLSTM1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    historyLSTM = modelLSTM1.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_test, y_test))

    #complie GRU
    modelGRU1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    historyGRU = modelGRU1.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_test, y_test))
    #fill in the data
    #LSTM train and test accuracy take 1
    #LSTM train and test loss take 1

    rowData["LSTM train accuracy take 1"] = historyLSTM.history["accuracy"][-1]
    rowData["LSTM test accuracy take 1"] = historyLSTM.history["val_accuracy"][-1]
    rowData["LSTM train loss take 1"] = historyLSTM.history["loss"][-1]
    rowData["LSTM test loss take 1"] = historyLSTM.history["val_loss"][-1]

    rowData["GRU train accuracy take 1"] = historyGRU.history["accuracy"][-1]
    rowData["GRU test accuracy take 1"] = historyGRU.history["val_accuracy"][-1]
    rowData["GRU train loss take 1"] = historyGRU.history["loss"][-1]
    rowData["GRU test loss take 1"] = historyGRU.history["val_loss"][-1]


    #FOR THE TAKE 2s

    # LSTM
    modelLSTM2 = Sequential()
    modelLSTM2.add(LSTM(hiddenUnit, input_shape=(x_train.shape[1], x_train.shape[2])))
    modelLSTM2.add(Dense(len(dictionary), activation="softmax"))
    print(modelLSTM2.summary())

    # GRU
    modelGRU2 = Sequential()
    modelGRU2.add(GRU(hiddenUnit, input_shape=(x_train.shape[1], x_train.shape[2]), reset_after=False))
    modelGRU2.add(Dense(len(dictionary), activation="softmax"))
    print(modelGRU2.summary())
    #compile both again
    # compile LSTM
    modelLSTM2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    historyLSTM = modelLSTM2.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_test, y_test))

    # complie GRU
    modelGRU2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    historyGRU = modelGRU2.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_test, y_test))
    # fill in the data
    # LSTM train and test accuracy take 1
    # LSTM train and test loss take 1

    rowData["LSTM train accuracy take 2"] = historyLSTM.history["accuracy"][-1]
    rowData["LSTM test accuracy take 2"] = historyLSTM.history["val_accuracy"][-1]
    rowData["LSTM train loss take 2"] = historyLSTM.history["loss"][-1]
    rowData["LSTM test loss take 2"] = historyLSTM.history["val_loss"][-1]

    rowData["GRU train accuracy take 2"] = historyGRU.history["accuracy"][-1]
    rowData["GRU test accuracy take 2"] = historyGRU.history3["val_accuracy"][-1]
    rowData["GRU train loss take 2"] = historyGRU.history["loss"][-1]
    rowData["GRU test loss take 2"] = historyGRU.history["val_loss"][-1]

    #leave the average for excel to do
    writer.writerow(rowData)
modelGRU2.save("Learned Models/learnedModelGRU.h5")
modelLSTM2.save("Learned Models/learnedModelLSTM.h5")
file.close()