import numpy as np
import tensorflow as tf
from numpy import array
from pickle import load
from tensorflow.python.keras import Input
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, LSTM, GRU
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.saving.save import load_model
import matplotlib.pyplot as plt

#Gates.py uses methods from Keras to access the inner workings of the LSTM and GRU networks and to investigate the behaviour of the gates
#Using the raw code for the GRU and LSTM network, the getGates methods store the activation values for the networks after they make each character predictions
#After a certain amount of predictions are made, a plot is created to show the spread of this activations and to give insight into the behaviour of the gates

arrayShape = open("Text Files and Dictionary/arrayShape.txt", "r")
shape = arrayShape.read().split("\n")


xTrainShape = tuple(list(map(int, shape[0].split(" "))))

modelLSTM = load_model("Learned Models/learnedModelLSTM.h5")
modelGRU = load_model("Learned Models/learnedModelGRU.h5")
dictionary = load(open("Text Files and Dictionary/dictionarySP.pkl", "rb"))

hidden_units = 90

#LSTM functional API definition
inputLSTM = Input(shape= (xTrainShape[1], xTrainShape[2]))
lstm1, lstm_state_h, state_c = LSTM(hidden_units, return_state=True)(inputLSTM)
denseLSTM = Dense(len(dictionary), activation="softmax")(lstm1)
modelLSTMFunc = Model(inputs=inputLSTM, outputs=[denseLSTM, lstm_state_h, state_c])

#set the weights from the trained LSTM network
for layer in modelLSTMFunc.layers:
    for layer1 in modelLSTM.layers:
            if layer.name == layer1.name:
                layer.set_weights(layer1.get_weights())


for layer in modelLSTMFunc.layers:
     if "LSTM" in str(layer):
         weightLSTMFunc = layer.get_weights()

for layer in modelLSTMFunc.layers:
    if "Dense" in str(layer):
        weightsDenseLSTM = layer.get_weights()


#GRU functional API definition
inputGRU = Input(shape= (xTrainShape[1], xTrainShape[2]))
gru1, gru_state_h= GRU(hidden_units, return_state=True, reset_after= False)(inputGRU)
denseGRU = Dense(len(dictionary), activation="softmax")(gru1)
modelGRUFunc = Model(inputs=inputGRU, outputs=[denseGRU, gru_state_h])

#set the weights from the trained GRU network
for layer in modelGRUFunc.layers:
    for layer1 in modelGRU.layers:
            if layer.name == layer1.name:
                layer.set_weights(layer1.get_weights())

            elif "dense" in layer.name and "dense" in layer1.name:
                layer.set_weights(layer1.get_weights())

for layer in modelGRUFunc.layers:
     if "GRU" in str(layer):
         weightGRUFunc = layer.get_weights()

for layer in modelGRUFunc.layers:
    if "Dense" in str(layer):
        weightsDenseGRU = layer.get_weights()


#Define the sigmoid function for the activations
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#encode sequence and predict character method
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

# create a method to generate a sequence using the regular tensorflow LSTM and GRU networks

def generateSeqLSTM(model, seqLen, chars_no, text):
    currentText = text
    hidden_states = []
    for i in range(chars_no):
        #given a sample text, it should be encoded as integers using the dictionary
        encodedSeq = encode_sequence(seqLen,currentText)
        predictedNo, h_tm1, c_tm1 = model.predict(encodedSeq)
        predictedChar = np.argmax(predictedNo, axis=-1)
        outputChar = ""
        for char, index in dictionary.items():
            if index == predictedChar:
                outputChar = char
                break
        currentText += outputChar
    return currentText, h_tm1, c_tm1

# generate sequence GRU method

def generateSeqGRU(model, seqLen, chars_no, text):
    currentText = text
    hidden_states = []
    for i in range(chars_no):
        #given a sample text, it should be encoded as integers using the dictionary
        encodedSeq = encode_sequence(seqLen,currentText)
        predictedNo, h_tm1 = model.predict(encodedSeq)
        predictedChar = np.argmax(predictedNo, axis=-1)
        outputChar = ""
        for char, index in dictionary.items():
            if index == predictedChar:
                outputChar = char
                break
        currentText += outputChar
    return currentText, h_tm1

#hand program the gates to predict sequences, given the weights
#GATES methods:

#LSTM:
def get_gatesLSTM(weight, x_t, h_tm1, c_tm1):
    # h_t shape : (1, 38)
    #x_t shape: (1, 38)
    #weight: warr, uarr, barr

    warr, uarr, barr = weight
    s_t = (x_t.dot(warr) + h_tm1.dot(uarr) + barr)

    hunit = uarr.shape[0]

    i = sigmoid(s_t[:, :hunit])            #input gate
    f = sigmoid(s_t[:, 1*hunit: 2*hunit])  #forget gate
    _c = np.tanh(s_t[:, 2*hunit: 3*hunit]) #candidate cell state
    o = sigmoid(s_t[:, 3*hunit:])           #output gate
    c_t = i*_c + f*c_tm1                    # cell state
    h_t = o*np.tanh(c_t)                    # hidden state

    return h_t, c_t, f, i, o

def get_gatesGRU(weight, x_t, h_tm1):
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
    return h_t, z, r


#DEFINE dense layers and their weights:
# LSTM:
inputResultLSTM = Input(shape = (1, hidden_units))
denseResultLSTM = Dense(len(dictionary), activation="softmax")(inputResultLSTM)
resultModelLSTM = Model(inputs= inputResultLSTM, outputs = denseResultLSTM)

for layer in resultModelLSTM.layers:
    if "Dense" in str(layer):
        layer.set_weights(weightsDenseLSTM)

# GRU:
inputResultGRU = Input(shape = (1, hidden_units))
denseResultGRU = Dense(len(dictionary), activation="softmax")(inputResultGRU)
resultModelGRU = Model(inputs= inputResultGRU, outputs = denseResultGRU)

for layer in resultModelGRU.layers:
    if "Dense" in str(layer):
        layer.set_weights(weightsDenseGRU)


text_to_predict = "black"


sequenceLength = 10
h_tm1_gru = [[0]]
sequence =""
f_t = []
i_t = []
o_t = []
h_tlstm = []
z_t = []
r_t= []
h_tgru = []
startingChar = '''all about y'''
# startingChar = "dark and de"
predictionLength = 20

gruPrediction = startingChar
lstmPrediction = startingChar



for i in range(predictionLength):
    if (i==0):
        predictedTextGRU, h_tm1_gru = generateSeqGRU(modelGRUFunc, sequenceLength, 2, gruPrediction)
        gruPrediction= predictedTextGRU

        #print(predictedTextGRU)

        predictedTextLSTM, h_tm1_lstm, c_tm1_lstm = generateSeqLSTM(modelLSTMFunc, sequenceLength, 2, lstmPrediction)
        lstmPrediction= predictedTextLSTM
        # print(predictedTextLSTM)
        continue

    # print("   GRU prediction")
    h_tm1_gru, z , r = get_gatesGRU(weightGRUFunc, encode_sequence(1, gruPrediction[-1]), h_tm1_gru)
    predictedNo = resultModelGRU.predict(array(h_tm1_gru).reshape(1, 1, hidden_units))
    # if(index ==3): print (predictedNo)
    # print(predictedNo)
    gruPrediction += predictedChar(predictedNo)
    # print(predictedChar(predictedNo))

    # print("   LSTM prediction")
    h_tm1_lstm, c_tm1_lstm, f, i, o = get_gatesLSTM(weightLSTMFunc, encode_sequence(1, lstmPrediction[-1]), h_tm1_lstm, c_tm1_lstm)
    predictedNo = resultModelLSTM.predict(array(h_tm1_lstm).reshape(1, 1, hidden_units))
    lstmPrediction+= predictedChar(predictedNo)
    # print(predictedChar(predictedNo))

    f_t.append(f)
    o_t.append(o)
    i_t.append(i)
    z_t.append(z)
    r_t.append(r)
    h_tlstm.append(h_tm1_lstm)
    h_tgru.append(h_tm1_gru)


print(gruPrediction)
print("\n")
print(lstmPrediction)




print("From GRU model")
something , h_t2 = generateSeqGRU(modelGRUFunc, sequenceLength, predictionLength, startingChar)
print(something)
print("\n")
print("lstm model results")
something , h_t2, c_t2 = generateSeqLSTM(modelLSTMFunc, sequenceLength, predictionLength, startingChar)
print(something)

#always check the original models to ensure the results from the gate functions are the same or similar

def saturation(f_t):
    #f_t represents the vector of any gate
    f_t = array(f_t)
    leftSaturation = []
    rightSaturation = []
    #reshape the vector into an acceptable shape
    f_t = f_t.reshape(f_t.shape[0],f_t.shape[2])

    #iterate through all the hidden units
    #if the value is less than 0.2, increase left saturation(LS) count
    #if the value is greater than 0.8, increase right saturation(RS) count
    for index, x in enumerate(f_t):
        if (index == 0):
            for i in x:
                leftSaturation.append(0)
                rightSaturation.append(0)
        for i, val in enumerate(x):
            if (val <= 0.2):
                leftSaturation[i] = leftSaturation[i] + 1
            elif (val >= 0.8):
                rightSaturation[i] = rightSaturation[i] + 1

#when the entire array has been iterated,
#check divide the number of LS counts likewise RS counts by the total length of
#vectors(predictions) to give the fraction of left&right saturation
    for index, x in enumerate(leftSaturation):
        total = array(f_t).shape[0]
        leftSaturation[index] = x / total
    for index, x in enumerate(rightSaturation):
        total = array(f_t).shape[0]
        rightSaturation[index] = x / total
    return leftSaturation, rightSaturation


leftSaturation_f, rightSaturation_f = saturation(f_t)
leftSaturation_i, rightSaturation_i = saturation(i_t)
leftSaturation_o, rightSaturation_o = saturation(o_t)
leftSaturation_z, rightSaturation_z = saturation(z_t)
leftSaturation_r, rightSaturation_r = saturation(r_t)
leftSaturation_h_tlstm, rightSaturation_h_tlstm = saturation(h_tlstm)
leftSaturation_h_tgru, rightSaturation_h_tgru = saturation(h_tgru)

print("F saturations")
print(leftSaturation_f)
print(rightSaturation_f)

print("I saturations")
print(leftSaturation_i)
print(rightSaturation_i)

print("O saturations")
print(leftSaturation_o)
print(rightSaturation_o)

print("Z saturations")
print(leftSaturation_z)
print(rightSaturation_z)

print("R saturations")
print(leftSaturation_r)
print(rightSaturation_r)

print("HTLSTM saturations")
print(leftSaturation_h_tlstm)
print(rightSaturation_h_tgru)

fig, ax = plt.subplots()

def plot(leftSaturation, rightSaturation, color, label,scale, n, ax):
    # n = hidden_units
    # scale = 200
    ax.scatter(leftSaturation, rightSaturation, c=color, s=scale, label = label,
               alpha = 0.1)



#plot(leftSaturation_f, rightSaturation_f, "blue", "forget",200, hidden_units, ax)
#plot(leftSaturation_i, rightSaturation_i, "red", "input",200, hidden_units, ax)
#plot(leftSaturation_o, rightSaturation_o, "orange", "output",200, hidden_units, ax)
#plot(leftSaturation_z, rightSaturation_z, "purple",  "update", 200,hidden_units, ax)
plot(leftSaturation_r, rightSaturation_r, "yellow", "reset", 200, hidden_units, ax)
#plot(leftSaturation_h_tlstm, rightSaturation_h_tlstm, "orange","hiddenLSTM", 200, hidden_units, ax)
#plot(leftSaturation_h_tgru, rightSaturation_h_tgru, "pink", "hiddenGRU",200, hidden_units, ax)

ax.legend()
ax.grid(True)
plt.show()

