import numpy as np
import tensorflow as tf
from numpy import array
from pickle import load
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.saving.save import load_model


modelLSTM = load_model("Learned Models/learnedModelLSTM.h5")
modelGRU = load_model("Learned Models/learnedModelGRU.h5")
modelLSTMFunc = load_model("Learned Models/learnedLSTMFunc.h5")
dictionary = load(open("Text Files and Dictionary/dictionarySP.pkl", "rb"))

def max(array):
    max = array[0][0]
    for x in array[0]:
        if x > max:
            max = x
    return max

def generateSeq(model, seqLen, chars_no, text):
    currentText = text
    hidden_states = []
    for i in range(chars_no):
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
    return currentText, array(encodedSeq).shape, hidden_states, predictedShape, predictedChar

#here it returns the predicted array for the last text. The prediction returns a onehot encoding array which picks a value closest to 1 i think

# historyLSTM = model.fit(x_train, y_train, epochs= 120, verbose=2, validation_data=(x_test, y_test))
# results = model.evaluate(x_test, y_test)
# print("test loss, test acc:", results)
# print(history.history)
#
# history = model.fit(x_train, y_train, epochs= 120, verbose=2, validation_data=(x_test, y_test))
# results = model.evaluate(x_test, y_test)
# print("test loss, test acc:", results)
# print(history.history)

print('LSTM predictions')
print(generateSeq(modelLSTM, 100, 1,'''The maid was in the garden, Hanging out the clothes, 
When down came a blackbird And pecked off her no'''))
print(" \n")
# print(generateSeq(modelLSTM, 100, 2,'''was opened The birds began to sing;
# Wasn't that a dainty dish, To set before the king. The king was i'''))
# print(" \n")
# print(generateSeq(modelLSTM, 100, 2,'''e parlour, Eating bread and honey. The maid was in the garden,
# When Four and Twenty king in the garde'''))
#print(generateSeq(modelLSTM, 100, 10,"maid was out in"))

print("GRU predictions")

print(generateSeq(modelGRU, 100, 2,'''The maid was in the garden, Hanging out the clothes, 
When down came a blackbird And pecked off her no'''))
print(" \n")
# print(generateSeq(modelGRU, 100, 2,'''was opened The birds began to sing;
# Wasn't that a dainty dish, To set before the king. The king was i'''))
# print(" \n")
# print(generateSeq(modelGRU, 100, 10,'''e parlour, Eating bread and honey. The maid was in the garden,
# When Four and Twenty king in the garde'''))
# print(generateSeq(modelGRU, 100, 10,"Sing a song of "))
# print(generateSeq(modelGRU, 100, 10,"A pocket full o"))
# print(generateSeq(modelGRU, 100, 10,"maid was in the"))
# print(generateSeq(modelGRU, 100, 10,"maid was out in"))

print('LSTM Functional API predictions')
def generateSeqFunc(model, seqLen, chars_no, text):
    currentText = text
    hidden_states = []
    for i in range(chars_no):
        #given a sample text, it should be encoded as integers using the dictionary
        encodedSeq = [dictionary[char] for char in currentText]
        #seperate the sequence into the groups of a certain length. Preferrably the number
        #of timesteps for the model/ maxLen here is the number of timesteps
        encodedSeq = pad_sequences([encodedSeq], maxlen=seqLen, truncating="pre")
        #onehot encode each of the chars in each element of the sequence
        encodedSeq = to_categorical(encodedSeq, num_classes=len(dictionary))
        #predict the character by running it through the learned model
        #predictedChar = model.predict_classes(encodedSeq, verbose=0)
        print(model.predict(encodedSeq))


generateSeqFunc(modelLSTMFunc, 100, 1, '''The maid was in the garden, Hanging out the clothes, 
When down came a blackbird And pecked off her no''')



#Create the formulas to calculate the cell states and the gates for GRU and LSTM
for layer in modelLSTM.layers:
        if "LSTM" in str(layer):
            weightLSTM = layer.get_weights()
            print("LSTM weights")
            #print(weightLSTM)
            warr, uarr , barr = weightLSTM
            print("Input weight")
            print(warr)
            print(warr.shape)
            print("Output weight")
            print(uarr)
            print(uarr.shape)
            print("Bias weight")
            print(barr)
            print(barr.shape)


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
    print("Shape of s_t" + s_t.shape)
    hunit = uarr.shape[0]
    i = sigmoid(s_t[:, :hunit])
    print("Shape of i" + i.shape)
    f = sigmoid(s_t[:, 1*hunit: 2*hunit])
    print("Shape of f" + f.shape)
    _c = np.tanh(s_t[:, 2*hunit: 3*hunit])
    print("Shape of _c" + _c.shape)
    o = sigmoid(s_t[:, 3*hunit:])
    print("Shape of o" + o.shape)
    c_t = i*_c + f*c_tm1
    print("Shape of c_t" + c_t.shape)
    h_t = o*np.tanh(c_t)
    print("Shape of h_t" + h_t.shape)
    return h_t, c_t

