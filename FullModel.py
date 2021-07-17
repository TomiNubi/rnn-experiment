import numpy as np
import tensorflow as tf
from numpy import array
from pickle import dump
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import GRU, LSTM, Dense


#define methods to load & save text files
def load_data(data):
    file = open(data, "r")
    text = file.read()
    file.close()
    return text
def save_data(data, filename):
    file = open(filename, "w")
    file.write(data)
    file.close


#define method to split a text file into sequences of certain lengths
def create_sequence(loadFile, seqLen, saveFile):

    text = load_data(loadFile)
    textLines = text.split()
    joinedText = " ".join(textLines)

    sequence = list()
    for i in range(seqLen, len(joinedText)):
        seq = joinedText[i - seqLen : i+1]
        sequence.append(seq)

    data_to_save = "\n".join(sequence)
    save_data(data_to_save, saveFile)

#define method to load sequences from file to array
def load_sequence(file):
    sequenceData = load_data(file)
    sequenceList = sequenceData.split("\n")
    return sequenceData, sequenceList

#create sequences for the three datasets
sequenceLength = 10

create_sequence("Text Files and Dictionary/Texts/If.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/If sequence.txt")
create_sequence("Text Files and Dictionary/Texts/IfTest.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/IfTest sequence.txt")

create_sequence("Text Files and Dictionary/Texts/Woods.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/Woods sequence.txt")
create_sequence("Text Files and Dictionary/Texts/WoodsTest.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/WoodsTest sequence.txt")

create_sequence("Text Files and Dictionary/Texts/1984.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/1984 sequence.txt")
create_sequence("Text Files and Dictionary/Texts/1984Test.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/1984Test sequence.txt")



#load the created train and test sequences
train_sequenceData, train_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/Woods sequence.txt")
test_sequenceData, test_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/WoodsTest sequence.txt")

#Determine the number of unique characters in the text, hence vocabulary size
charSet = sorted(set(train_sequenceData))
dictionary = dict((char, index) for index, char in enumerate(charSet))
vocabSize = len(dictionary)

#encode the sequences into integers
def int_encode(sequenceList):
    integer_sequence = list()
    for line in sequenceList:
        seq = [dictionary[char] for char in line]
        integer_sequence.append(seq)

    '''next we need to one hot encode each sequence such that it is a 
    0-1 vector with length of the vocabulary size
    with a 1 representing the index where the character exists.'''

    integer_sequence = array(integer_sequence)
    x, y = integer_sequence[:, :-1], integer_sequence[:, -1]
    int_sequences = [to_categorical(num, num_classes=vocabSize) for num in x]
    x = array(int_sequences)
    y= to_categorical(y, num_classes=vocabSize)
    return x, y

x_train , y_train = int_encode(train_sequenceList)
x_test , y_test  = int_encode(test_sequenceList)

#Using the Tensorflow library, create the networks

#LSTM
#configure LSTM model
modelLSTM = Sequential()
modelLSTM.add(LSTM (90, input_shape=(x_train.shape[1], x_train.shape[2])))
modelLSTM.add(Dense (len(dictionary), activation="softmax"))
print(modelLSTM.summary())

#compile & train LSTM model
modelLSTM.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = modelLSTM.fit(x_train, y_train, epochs= 100, verbose=2, validation_data=(x_test, y_test))
results = modelLSTM.evaluate(x_test, y_test)
print("test loss, test acc:", results)


#GRU
#configure GRU model
modelGRU = Sequential()
modelGRU.add(GRU (90, input_shape=(x_train.shape[1], x_train.shape[2]), reset_after=False))
modelGRU.add(Dense (len(dictionary), activation="softmax"))
print(modelGRU.summary())

#compile & train GRU model
modelGRU.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = modelGRU.fit(x_train, y_train, epochs= 100, verbose=2, validation_data=(x_test, y_test))
results = modelGRU.evaluate(x_test, y_test)
print("test loss, test acc:", results)