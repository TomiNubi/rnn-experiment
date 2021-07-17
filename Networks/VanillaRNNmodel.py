import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, SimpleRNN
from tensorflow.python.keras.layers import Dense
from pickle import dump, load

#first step is to load the data
from tensorflow.python.keras.models import load_model


def load_data(data):
    file = open(data, "r")
    text = file.read()
    file.close()
    return text
def save_data(data, filename):
    file = open(filename, "w")
    file.write(data)
    file.close

text = load_data("Sixpence.txt")
textLines = text.split()
joinedText = " ".join(textLines)

#Create the sequence of data that is going to be feed into the RNN. These would later be
#seperated into input and output vectors
#What are the three dimensional inputs like?

seqLen = 10 #toggle between different values
sequence = list()
for i in range(seqLen, len(joinedText)):
    seq = joinedText[i - seqLen : i+1]
    sequence.append(seq)

print("Length of sequence is: %d " % len(sequence))

data_to_save = "\n".join(sequence)
save_data(data_to_save, "../Extras/Sixpence sequence2.txt")

#encode the sequences into integers
#First we need to determine the vocabulary size for the tetx. How many unique characters are there

#load the sequence file
sequenceData = load_data("Sixpence sequence2.txt")
sequenceList = sequenceData.split("\n")
print(sequenceData)
#filter all the unique characters by putting it into a set

charSet = sorted(set(sequenceData))
dictionary = dict((char, index) for index, char in enumerate(charSet))
vocabSize = len(dictionary)

#We can integer encode each line in the sequence now as:
integer_sequence = list()
for line in sequenceList:
    seq = [dictionary[char] for char in line]
    integer_sequence.append(seq)
#print(integer_sequence)

#next we need to one hot encode each sequence such that it is a 0-1 vector with length of the vocabulary size
#with a one representing the point where the character exists.
integer_sequence = array(integer_sequence)
x, y = integer_sequence[:, :-1], integer_sequence[:, -1]
#print(x)
int_sequences = [to_categorical(num, num_classes=vocabSize) for num in x]
x = array(int_sequences)
y= to_categorical(y, num_classes=vocabSize)


#create the RNN model
model = Sequential()
model.add(SimpleRNN(76, input_shape=(x.shape[1], x.shape[2])))
model.add(Dense (vocabSize, activation="tanh"))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x, y, epochs= 120, verbose=2)

model.save("learnedModelSP2.h5")
dump(dictionary, open("dictionarySP2.pkl", "wb"))

newmodel = load_model("learnedModelSP2.h5")
newdictionary = load(open("dictionarySP2.pkl", "rb"))

def generateSeq(model, seqLen, chars_no, text):
    currentText = text
    for i in range(chars_no):
        #given a sample text, it should be encoded as integers using the dictionary
        encodedSeq = [newdictionary[char] for char in currentText]
        #seperate the sequence into the groups of a certain length. Preferrably the number
        #of timesteps for the model/ maxLen here is the number of timesteps
        encodedSeq = pad_sequences([encodedSeq], maxlen=seqLen, truncating="pre")
        #onehot encode each of the chars in each element of the sequence
        encodedSeq = to_categorical(encodedSeq, num_classes=len(newdictionary))
        #predict the character by running it through the learned model
        #predictedChar = model.predict_classes(encodedSeq, verbose=0)
        predictedChar = np.argmax(model.predict(encodedSeq), axis=-1)
        #print(predictedChar)
        #once the character has been predicted, reverse it back to a word
        outputChar = ""
        for char, index in newdictionary.items():
            if index == predictedChar:
                outputChar = char
                break
        currentText += outputChar
    return currentText


print(generateSeq(newmodel, 10, 20,"Sing a son"))
print(generateSeq(newmodel, 10, 20,"A pocket f"))
print(generateSeq(newmodel, 10, 20,"maid was i"))
print(generateSeq(newmodel, 10, 20,"maid was o"))
