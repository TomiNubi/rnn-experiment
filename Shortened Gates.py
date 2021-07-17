import numpy as np
import tensorflow as tf
from numpy import array
from pickle import load
from tensorflow.python.keras import Input
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

dictionary = load(open("Text Files and Dictionary/dictionarySP.pkl", "rb"))


'''
given already trained LSTM and GRU networks:

LSTM:
    name: modelLSTM
    weights: weightLSTM
GRU:
    name:modelGRU
    weights: weightLSTM

given a
'''

#Define the sigmoid function for the activations
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#hand program the gates to predict sequences, given the weights
#GATES methods:

#LSTM:
def get_gatesLSTM(weight, x_t, h_tm1, c_tm1):
    # h_t shape : (1, 90)
    #x_t shape: (1, 38)
    #i, o, f, _c shape: (1, 90)
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
    # h_t shape : (1, 90)
    #x_t shape: (1, 38)
    #z, r and _h shape: (1, 90)
    #weight: warr, uarr, barr

    warr, uarr, barr = weight
    hunit = uarr.shape[0]
    x_z = x_t.dot(warr[:, :hunit]) + barr[ :hunit]
    x_r = x_t.dot(warr[:, hunit: 2*hunit]) + barr[hunit: 2*hunit]
    x_h = x_t.dot(warr[:,2*hunit:]) + barr[2*hunit:]

    recurrent_z = h_tm1.dot(uarr[:, :hunit])
    recurrent_r = h_tm1.dot(uarr[:, hunit: 2*hunit])

    z = sigmoid(x_z + recurrent_z)  #update gate
    r = sigmoid(x_r + recurrent_r)  #reset gate

    recurrent_h = (h_tm1 * r).dot(uarr[:,2*hunit:])
    _h = np.tanh(x_h + recurrent_h)  #candidate hidden state

    h_t = z*h_tm1 + (1-z)*_h
    return h_t, z, r




sequenceLength = 10

#initialise arrays to store the values of the gates during each prediction in
f_t = []
i_t = []
o_t = []
h_tlstm = []
z_t = []
r_t= []
h_tgru = []

startingChar = '''all about y'''
predictionLength = 20

gruPrediction = startingChar
lstmPrediction = startingChar


#loop for the specified nnumber of predicted lengths and generate the clculated hidden units using the getgates functions
for i in range(predictionLength):
    #for the first index, use the pre-made Keras networks, to print out the first hidden state to be used by the networks
    if (i==0):
        predictedTextGRU, h_tm1_gru = generateSeqGRU(modelGRU, sequenceLength, 2, gruPrediction)
        gruPrediction= predictedTextGRU

        predictedTextLSTM, h_tm1_lstm, c_tm1_lstm = generateSeqLSTM(modelLSTM, sequenceLength, 2, lstmPrediction)
        lstmPrediction= predictedTextLSTM
        continue

    #GRU prediction
    h_tm1_gru, z , r = get_gatesGRU(weightGRU, encode_sequence(1, gruPrediction[-1]), h_tm1_gru)

    #LSTM prediction
    h_tm1_lstm, c_tm1_lstm, f, i, o = get_gatesLSTM(weightLSTM, encode_sequence(1, lstmPrediction[-1]), h_tm1_lstm, c_tm1_lstm)
    
    f_t.append(f)
    o_t.append(o)
    i_t.append(i)
    z_t.append(z)
    r_t.append(r)
    h_tlstm.append(h_tm1_lstm)
    h_tgru.append(h_tm1_gru)


def saturation(f_t):
    #f_t represents an array containing a set of gate vectors
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

#when the entire array has been iterated,check divide the number of LS counts likewise RS counts by the total length of vectors(predictions) to give the fraction of left&right saturation
    for index, x in enumerate(leftSaturation):
        total = array(f_t).shape[0]
        leftSaturation[index] = x / total
    for index, x in enumerate(rightSaturation):
        total = array(f_t).shape[0]
        rightSaturation[index] = x / total
    return leftSaturation, rightSaturation

#get the saturation arrays for all of the gates
leftSaturation_f, rightSaturation_f = saturation(f_t)
leftSaturation_i, rightSaturation_i = saturation(i_t)
leftSaturation_o, rightSaturation_o = saturation(o_t)
leftSaturation_z, rightSaturation_z = saturation(z_t)
leftSaturation_r, rightSaturation_r = saturation(r_t)


fig, ax = plt.subplots()
#define a function to plot the arrays on a scatter plot with x axis representing left-saturation and y axis representing right saturation 
def plot(leftSaturation, rightSaturation, color, label,scale, ax):
    ax.scatter(leftSaturation, rightSaturation, c=color, s=scale, label = label, alpha = 0.1)

#plot the saturation graphs for all the gates
plot(leftSaturation_f, rightSaturation_f, "blue", "forget",200,  ax)
plot(leftSaturation_i, rightSaturation_i, "red", "input",200,  ax)
plot(leftSaturation_o, rightSaturation_o, "orange", "output",200,  ax)
plot(leftSaturation_z, rightSaturation_z, "purple",  "update", 200, ax)
plot(leftSaturation_r, rightSaturation_r, "yellow", "reset", 200,  ax)

ax.legend()
ax.grid(True)
plt.show()

