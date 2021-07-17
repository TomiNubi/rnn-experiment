import numpy as np
import tensorflow as tf
from numpy import array
from pickle import dump
from tensorflow.keras.utils import to_categorical


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

#encode the sequences into integers
#First we need to determine the vocabulary size for the tetx. How many unique characters are there
sequenceLength = 20


# create_sequence("Text Files and Dictionary/Sixpence.txt", sequenceLength, "Text Files and Dictionary/Sixpence sequence.txt")
# create_sequence("Text Files and Dictionary/testData.txt", sequenceLength, "Text Files and Dictionary/testData sequence.txt")

create_sequence("Text Files and Dictionary/Texts/If.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/If sequence.txt")
create_sequence("Text Files and Dictionary/Texts/IfTest.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/IfTest sequence.txt")

create_sequence("Text Files and Dictionary/Texts/Woods.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/Woods sequence.txt")
create_sequence("Text Files and Dictionary/Texts/WoodsTest.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/WoodsTest sequence.txt")

create_sequence("Text Files and Dictionary/Texts/1984.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/1984 sequence.txt")
create_sequence("Text Files and Dictionary/Texts/1984Test.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/1984Test sequence.txt")

# create_sequence("Text Files and Dictionary/Texts/long.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/long sequence.txt")
# create_sequence("Text Files and Dictionary/Texts/longTest.txt.txt", sequenceLength, "Text Files and Dictionary/Texts/Sequences/longTest.txt sequence.txt")
#load the sequence file
def load_sequence(file):
    sequenceData = load_data(file)
    sequenceList = sequenceData.split("\n")
    return sequenceData, sequenceList

# train_sequenceData, train_sequenceList = load_sequence("Text Files and Dictionary/Sixpence sequence.txt")
# test_sequenceData, test_sequenceList = load_sequence("Text Files and Dictionary/testData sequence.txt")

train_sequenceData, train_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/Woods sequence.txt")
test_sequenceData, test_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/WoodsTest sequence.txt")

# train_sequenceData, train_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/If sequence.txt")
# test_sequenceData, test_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/IfTest sequence.txt")

# train_sequenceData, train_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/1984 sequence.txt")
# test_sequenceData, test_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/1984Test sequence.txt")

# train_sequenceData, train_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/long sequence.txt")
# test_sequenceData, test_sequenceList = load_sequence("Text Files and Dictionary/Texts/Sequences/longTest sequence.txt")

#create a dictionary by filtering all the unique characters in the training set by putting it into a set
charSet = sorted(set(train_sequenceData))
dictionary = dict((char, index) for index, char in enumerate(charSet))
vocabSize = len(dictionary)

#We can integer encode each line in the sequence now as:
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
    int_sequences = [to_categorical(num, num_classes=vocabSize) for num in x]
    x = array(int_sequences)
    y= to_categorical(y, num_classes=vocabSize)
    return x, y

train_x , train_y = int_encode(train_sequenceList)
test_x , test_y = int_encode(test_sequenceList)
#Here we prepare the test data, I'm separating it so that I can manually tweak the test data for the sake of experimentation


arrayShape = open("Text Files and Dictionary/arrayShape.txt", "w")
arrayShape.write(" ".join(str(x) for x in train_x.shape))
arrayShape.write("\n")
arrayShape.write(" ".join(str(y) for y in train_y.shape))
arrayShape.write("\n")
arrayShape.write(" ".join(str(x) for x in test_x.shape))
arrayShape.write("\n")
arrayShape.write(" ".join(str(y) for y in test_y.shape))

np.array(train_x).tofile("Text Files and Dictionary/xTrain.bin")
np.array(train_y).tofile("Text Files and Dictionary/yTrain.bin")
np.array(test_x).tofile("Text Files and Dictionary/xTest.bin")
np.array(test_y).tofile("Text Files and Dictionary/yTest.bin")

print("Vocab size is " + str(vocabSize))
print(dictionary)
dump(dictionary, open("Text Files and Dictionary/dictionarySP.pkl", "wb"))