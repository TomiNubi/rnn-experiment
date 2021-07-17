import numpy as np
import tensorflow as tf
from numpy import array
from pickle import load
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, LSTM, GRU
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.saving.save import load_model

arrayShape = open("Text Files and Dictionary/arrayShape.txt", "r")
shape = arrayShape.read().split("\n")


xTrainShape = tuple(list(map(int, shape[0].split(" "))))



modelLSTM = load_model("Learned Models/learnedModelLSTM.h5")
modelGRU = load_model("Learned Models/learnedModelGRU.h5")
#modelLSTMFunc = load_model("Learned Models/learnedLSTMFunc.h5")
dictionary = load(open("Text Files and Dictionary/dictionarySP.pkl", "rb"))

hidden_units = 90

input = Input(shape= (xTrainShape[1], xTrainShape[2]))
gru1, state_h= GRU(hidden_units, return_state=True, reset_after= False)(input)
dense = Dense(len(dictionary), activation="softmax")(gru1)
modelGRUFunc = Model(inputs=input, outputs=[dense, state_h])



for layer in modelGRUFunc.layers:
    for layer1 in modelGRU.layers:
            if layer.name == layer1.name:
                layer.set_weights(layer1.get_weights())
                print("Weights set")

for layer in modelGRU.layers:
     if "GRU" in str(layer):
         weightGRUNorm = layer.get_weights()

for layer in modelGRUFunc.layers:
     if "GRU" in str(layer):
         weightGRUFunc = layer.get_weights()

for layer in modelGRUFunc.layers:
    if "Dense" in str(layer):
        weightsDenseFunc = layer.get_weights()

warr, uarr, barr = weightGRUFunc
print(warr.shape, uarr.shape, barr.shape)
warr, uarr, barr = weightGRUNorm
print(warr.shape, uarr.shape, barr.shape)
print('LSTM Functional API predictions')
#print(barr)



def generateSeq(model, seqLen, chars_no, text):
    currentText = text
    hidden_states = []
    for i in range(chars_no):
        print("Iterated")
        #given a sample text, it should be encoded as integers using the dictionary
        encodedSeq = [dictionary[char] for char in currentText]
        #seperate the sequence into the groups of a certain length. Preferrably the number
        #of timesteps for the model/ maxLen here is the number of timesteps
        encodedSeq = pad_sequences([encodedSeq], maxlen=seqLen, truncating="pre")
        #onehot encode each of the chars in each element of the sequence
        encodedSeq = to_categorical(encodedSeq, num_classes=len(dictionary))
        #predict the character by running it through the learned model
        #predictedChar = model.predict_classes(encodedSeq, verbose=0)
        predictedNo= model.predict(encodedSeq)
        hidden_states.append(predictedNo)
        predictedShape = array(predictedNo).shape
        predictedChar = np.argmax(predictedNo, axis=-1)
        maxNo = max(predictedNo)
        #print(predictedChar)
        #once the character has been predicted, reverse it back to a word
        outputChar = ""
        for char, index in dictionary.items():
            if index == predictedChar:
                outputChar = char
                break
        currentText += outputChar
    return currentText, hidden_states

def encode_sequence(seqLen, currentText):
    encodedSeq = [dictionary[char] for char in currentText]
    # seperate the sequence into the groups of a certain length. Preferrably the number
    # of timesteps for the model/ maxLen here is the number of timesteps
    encodedSeq = pad_sequences([encodedSeq], maxlen=seqLen, truncating="pre")
    # onehot encode each of the chars in each element of the sequence
    encodedSeq = to_categorical(encodedSeq, num_classes=len(dictionary))
    return encodedSeq


def generateSeqFunc(model, seqLen, chars_no, text):
    currentText = text
    hidden_states = []
    for i in range(chars_no):
        #given a sample text, it should be encoded as integers using the dictionary
        encodedSeq = encode_sequence(seqLen,currentText)
        print(encodedSeq)
        #print(model.predict(encodedSeq))
        predictedNo, h_tm1 = model.predict(encodedSeq)
        predictedChar = np.argmax(predictedNo, axis=-1)
        outputChar = ""
        for char, index in dictionary.items():
            if index == predictedChar:
                outputChar = char
                break
        currentText += outputChar
    return currentText, h_tm1


#Create the formulas to calculate the cell states and the gates for GRU and LSTM

#First define the sigmoid function

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


#to calculate the weights, i need to get the weights biases


def get_gates(weight, x_t, h_tm1):
    # h_t shape : (1, 38)
    #x_t shape: (1, 38)
    #weight: warr, uarr, barr

    warr, uarr, barr = weight
    hunit = uarr.shape[0]
    x_z = x_t.dot(warr[:, :hunit]) + barr[ :hunit]
    x_r = x_t.dot(warr[:, hunit: 2*hunit]) + barr[hunit: 2*hunit]
    x_h = x_t.dot(warr[:,2*hunit:]) + barr[2*hunit:]

    recurrent_z = h_tm1.dot(uarr[:, :hunit])
    recurrent_r = h_tm1.dot(uarr[:, hunit: 2*hunit])

    z = sigmoid(x_z + recurrent_z)
    r = sigmoid(x_r + recurrent_r)

    recurrent_h = (h_tm1 * r).dot(uarr[:,2*hunit:])
    _h = np.tanh(x_h + recurrent_h)

    h_t = z*h_tm1 + (1-z)*_h
    return h_t


def predictedChar(predicted_No):
    predictedChar = np.argmax(predicted_No, axis=-1)
    outputChar = ""
    for char, index in dictionary.items():
        if index == predictedChar:
            outputChar = char
            break
    return outputChar


# text_to_predict = "Hangi"
# predictedText, h_tm1= generateSeqFunc(modelGRUFunc, 5, 1, text_to_predict )
# h_t = get_gates(weightGRUFunc, encode_sequence(1, "n") , h_tm1)
# print("Get gates result")
# print(h_t)

inputResult = Input(shape = (1, hidden_units))
denseResult = Dense(len(dictionary), activation="softmax")(inputResult)
resultModel = Model(inputs= inputResult, outputs = denseResult)

for layer in resultModel.layers:
    if "Dense" in str(layer):
        layer.set_weights(weightsDenseFunc)

# print("After Dense")
# predictedNo = resultModel.predict(array(h_t).reshape(1, 1, hidden_units))
# print(predictedNo)
# print(predictedChar(predictedNo))
# #
#
# print("GRU model results")
# something , h_t2 = generateSeqFunc(modelGRUFunc, 5, 1, "Hangin")
# print(h_t2)
# print(something)


inputValue = "Hangi"
[_, hiddenValue]  = generateSeqFunc(modelGRUFunc, 5, 1, inputValue)
predictedNo = resultModel.predict(array(hiddenValue).reshape(1, 1, hidden_units))
newValue = inputValue + predictedChar(predictedNo)
print("The starting value for both functions is:" + newValue)
#if the function used are the same and so are the weights then calling this should give the same answer as get gates

#So now

#call get_gates for the GRU
gates_hidden= get_gates(weightGRUFunc, encode_sequence(1, inputValue[-1]), hiddenValue)

#call gru function
[_, func_hidden]  = generateSeqFunc(modelGRUFunc, 5, 1, newValue)

print(gates_hidden, "\n")

print(func_hidden)


