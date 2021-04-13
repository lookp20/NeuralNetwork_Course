# -*- coding: utf-8 -*-

import numpy as np
import nnwplot
import matplotlib.pyplot as plt

class SLN:
    def __init__(self,dIn,cOut):
        np.random.seed(42)
        self._dIn = dIn
        self._cOut = cOut
        self._W = np.random.randn(cOut,dIn) / np.sqrt(dIn+1)
        self._b = np.zeros(cOut)[np.newaxis].T
        # Aus Aufgabenblatt 1
        # self._W = np.array([[-.3, 1]])
        # self._b = np.array([0])
        
    def neuron(self, X):
        N = X.shape[1]
        net = np.zeros(N)
        threshold = 0
    
        for n in range(0,N): # iterate over instances
            for j in range(0,self._dIn): # iterate over weights
                 net[n] += self._W[0,j] * X[j, n]
            net[n] += self._b * 1
        return net > threshold
    
    def DeltaTrain(self,X,T,eta,maxIter,maxErrorRate):
        N = X.shape[1]
        plt.ion()
        for i in range(maxIter):
            Y = self.neuron(X) # classify
            err = ErrorRate(Y,T)  # calculate error rate
            print(err)
            if err < maxErrorRate:  # stop if maxErrorRate reached
                break
            deltaWkj = np.zeros(3)
            for j in range(self._dIn): # iterate over weights
                summe = 0
                for n in range(N): # iterate over input neurons
                    summe += eta*(T[n]-Y[n])*X[j,n]
                deltaWkj[j] = 1/N*summe
                
            # train bias neuron
            summe = 0
            for n in range(N):
                summe += eta*(T[n]-Y[n])*1
            deltaWkj[2] = 1/N*summe
            
            self._W[0,0] += deltaWkj[0]
            self._W[0,1] += deltaWkj[1]
            self._b[0,0] += deltaWkj[2]
            nnwplot.plotTwoFeatures(X[:2],T,self.neuron)
                
'''
Falls Y ein Vektor ist, dann ist die Fehlerrate gleich der Summe
der falsch-klassifizierten Instanzen dividiert durch die Anzahl der Instanzen.
Falls Y eine one-hot Matrix ist, dann werden die Indizes der Maximumwerte
verglichen, falsch-klassifizierte Instanzen aufsummiert und durch die Anzahl
der Instanzen dividiert.

One-hot Kodierung
1 0 0 1 ... -> argmax(0) -> [0, 1, 1, 0, ...] 
0 1 1 0 ...
'''
def ErrorRate(Y,T):
    if Y.ndim==1 or Y.shape[0]==1:
        errors=Y!=T
        return errors.sum()/Y.size
    else: # für mehrere Ausgaben in one-hot Kodierung:
        errors=Y.argmax(0)!=T.argmax(0)
        return errors.sum()/Y.shape[1]        


#%% Aufgabe 3e
iris_data = np.loadtxt('iris.csv', delimiter=',') # load iris dataset
X = iris_data[:100,:4].T # select first 100 instances with 4 features and transpose
T = iris_data[:100,4].T # select first 100 instances with label and transpose

m=SLN(2,1) # create SLN with 2 input neuron and 1 output neuron

m.DeltaTrain(X[:2],T,0.01,1000,0.05) # train SLN

#%% Aufgabe 3d
und_model = SLN(2,1)

train_data = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,1]])
X_und = train_data[:,:2].T
T_und = train_data[:,2].T

und_model.DeltaTrain(X_und,T_und,0.1,1000,0.02) # train SLN
#%% Aufgabe 3f
m = SLN(2,1)
m.DeltaTrain(X[1:3],T,0.01,1000,0.02) # train SLN