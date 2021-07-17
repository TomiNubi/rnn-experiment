import numpy as np
import tensorflow as tf
from numpy import array
from pickle import load
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.saving.save import load_model

arrayShape = open("Text Files and Dictionary/arrayShape.txt", "r")
shape = arrayShape.read().split("\n")


xTrainShape = tuple(list(map(int, shape[0].split(" "))))



modelLSTM = load_model("Learned Models/learnedModelLSTM.h5")
modelGRU = load_model("Learned Models/learnedModelGRU.h5")
#modelLSTMFunc = load_model("Learned Models/learnedLSTMFunc.h5")
dictionary = load(open("Text Files and Dictionary/dictionarySP.pkl", "rb"))

input = Input(shape= (xTrainShape[1], xTrainShape[2]))
lstm1, state_h, state_c = LSTM(90, return_state=True)(input)
dense = Dense(len(dictionary), activation="softmax")(lstm1)
modelLSTMFunc = Model(inputs=input, outputs=[dense, state_h, state_c])

for layer in modelLSTMFunc.layers:
    for layer1 in modelLSTM.layers:
            if layer.name == layer1.name:
                layer.set_weights(layer1.get_weights())
                print("Weights set")

for layer in modelLSTM.layers:
     if "LSTM" in str(layer):
         weightLSTMNorm = layer.get_weights()

for layer in modelLSTMFunc.layers:
     if "LSTM" in str(layer):
         weightLSTMFunc = layer.get_weights()

for layer in modelLSTMFunc.layers:
    if "Dense" in str(layer):
        weightsDenseFunc = layer.get_weights()



print('LSTM Functional API predictions')

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
# print(generateSeqFunc(modelLSTMFunc, 5, 1, "The m"))
# print("Normal LSTM")






#Create the formulas to calculate the cell states and the gates for GRU and LSTM


#First define the sigmoid function

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#to calculate the weights, i need to get the weights biases


#get_weights():


def get_gates(weight, x_t, h_tm1, c_tm1):
    # h_t shape : (1, 38)
    #x_t shape: (1, 38)
    #weight: warr, uarr, barr

    warr, uarr, barr = weight
    s_t = (x_t.dot(warr) + h_tm1.dot(uarr) + barr)

    hunit = uarr.shape[0]

    i = sigmoid(s_t[:, :hunit])
    f = sigmoid(s_t[:, 1*hunit: 2*hunit])
    _c = np.tanh(s_t[:, 2*hunit: 3*hunit])
    o = sigmoid(s_t[:, 3*hunit:])
    c_t = i*_c + f*c_tm1
    h_t = o*np.tanh(c_t)
    return h_t, c_t

def encode_sequence(seqLen, currentText):
    encodedSeq = [dictionary[char] for char in currentText]
    # seperate the sequence into the groups of a certain length. Preferrably the number
    # of timesteps for the model/ maxLen here is the number of timesteps
    encodedSeq = pad_sequences([encodedSeq], maxlen=seqLen, truncating="pre")
    # onehot encode each of the chars in each element of the sequence
    encodedSeq = to_categorical(encodedSeq, num_classes=len(dictionary))
    return encodedSeq

def predictedChar(predicted_No):
    predictedChar = np.argmax(predicted_No, axis=-1)
    outputChar = ""
    for char, index in dictionary.items():
        if index == predictedChar:
            outputChar = char
            break
    return outputChar

def generateSeqFunc(model, seqLen, chars_no, text):
    currentText = text
    hidden_states = []
    for i in range(chars_no):
        #given a sample text, it should be encoded as integers using the dictionary
        encodedSeq = encode_sequence(seqLen,currentText)
        print(model.predict(encodedSeq))
        predictedNo, h_tm1, c_tm1 = model.predict(encodedSeq)
        predictedChar = np.argmax(predictedNo, axis=-1)
        outputChar = ""
        for char, index in dictionary.items():
            if index == predictedChar:
                outputChar = char
                break
        currentText += outputChar
    return currentText, h_tm1, c_tm1



text_to_predict = "Hanging "

predictedText, h_tm1, c_tm1 = generateSeqFunc(modelLSTMFunc, 5, 1, text_to_predict )
# print("Functional Weights")
# print(weightLSTMFunc)
# print("Normal Weights")
# print(weightLSTMNorm)
h_t, c_t = get_gates(weightLSTMFunc, encode_sequence(1, predictedText[-1]) , h_tm1, c_tm1)
print(predictedText[-1])
print("Get gates result")
#print(h_t)

inputResult = Input(shape = h_t.shape)
denseResult = Dense(len(dictionary), activation="softmax")(inputResult)
resultModel = Model(inputs= inputResult, outputs = denseResult)

for layer in resultModel.layers:
    if "Dense" in str(layer):
        layer.set_weights(weightsDenseFunc)

print("After Dense")
predictedNo = resultModel.predict(array(h_t).reshape(1, 1, 90))
#print(predictedNo)
print(predictedChar(predictedNo))


print("lstm model results")
something , h_t2, c_t2 = generateSeqFunc(modelLSTMFunc, 5, 2, text_to_predict)
#print(h_t2)
print(something)

