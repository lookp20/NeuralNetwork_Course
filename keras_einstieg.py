# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:56:41 2021

Lösung zu Aufgabenblatt Nr 3
@author: NNW Gruppe 8
    Georgii Kostiuchik
    Look Phanthavong
    Tobias Brückner
"""

# Import libraries
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.utils import to_categorical,  plot_model
from keras import Input
from keras.callbacks import LambdaCallback
from matplotlib import pyplot as plt
import numpy as np
import nnwplot_2

# Load iris data
data = np.loadtxt("iris.csv", delimiter = ",")
X = data[:, 2:4]
T = to_categorical(data[:, 4])

#%% Aufgabe 1
model = Sequential()

# Erster Aufgabenteil
# model.add(Dense(3, input_shape = (2, ), activation = 'softmax'))

# Erweiterung (Aufgabenteil 1e)
model.add(Dense(5, input_shape = (2, ), activation = 'tanh'))
model.add(Dense(3, activation = 'softmax'))
model.compile(optimizer = 'adam',  loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X, T, epochs = 1000)

# Plot result
nnwplot_2.plotTwoFeatures(X, T, model.predict)

# Erweiterung: Es ergeben sich 33 Gewichte:
# Für die erste Schicht 2 Eingaben x 5 Hidden Neuronen + 5 Bias Gewichte => 15 Parameter
# Für die zweite Schicht 5 Eingaben x 3 Ausgabe Neuronen + 3 Bias Gewichte => 18 Parameter
# In Summe 15 + 18 = 33 Parameter/Gewichte
print(model.summary())

#%% Aufgabe 2
# Create model and compile it
inp = Input(2,)
l1 = Dense(5, activation = 'tanh')(inp)
l2 = Dense(3, activation = 'softmax')(l1)
model2 = Model(inp, l2)

model2.compile(optimizer = 'adam',  loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Method used for Lambda Callback for plotting the result after every ten epochs
def plotEveryTenEpochs(epoch, logs):
    if (epoch%10 == 0):
        nnwplot_2.plotTwoFeatures(X, T, model2.predict)

plot_callback = LambdaCallback(on_epoch_end = plotEveryTenEpochs)
# Fit the model
hist = model2.fit(X, T, epochs = 200, callbacks = [plot_callback])

nnwplot_2.plotTwoFeatures(X, T, model2.predict)

# Get summary of model with its layers and parameters
print(model2.summary())

# Plot model architecture
plot_model(model2, to_file = 'model.png')

losses = hist.history['loss']
accs = hist.history['accuracy']

# Plot history of loss and accuracy
fig, axs = plt.subplots(2)
fig.suptitle('Model Loss and Accuracy')
axs[0].plot(losses)
axs[0].set(xlabel='Epoch', ylabel='Loss')
axs[1].plot(accs)
axs[1].set(xlabel='Epoch', ylabel='Accuracy')
plt.show()


