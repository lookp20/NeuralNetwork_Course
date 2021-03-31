# -*- coding: utf-8 -*-
"""
Gruppenmitglieder: 
Tobias Brückner
Georgii Kostiuschik
Look Phanthavong
"""

import numpy as np
import matplotlib.pyplot as plt
import nnwplot

data = np.loadtxt("iris.csv", delimiter = ",")
# print(data.shape)

X = data[:, 0:4].T
# print(X.shape)

T = data[:, [4]].T
# print(T.shape)
plt.scatter(X[0,:], X[1,:], c = T, cmap = plt.cm.prism)

def neuron(X):
    N = X.shape[1]
    net = np.zeros(N)
    W = [-0.3, 1]
    threshold = 2
    
    for i in range(0,N):
        for j in range(0,2):
             net[i] += W[j] * X[0:2, i][j]
    return net > threshold

nnwplot.plotTwoFeatures(X[0:2], T, neuron)
    
    