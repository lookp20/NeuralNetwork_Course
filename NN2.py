# -*- coding: utf-8 -*-

import numpy as np
import nnwplot

np.random.seed(42)

class SLN:
    def __init__(self,dIn,cOut):
        self._dIn = dIn
        self._cOut = cOut
        self._W = np.random.randn(cOut,dIn) / np.sqrt(dIn+1)
        self._b = np.zeros(cOut)[np.newaxis].T
        # self._W = np.array([[-.3, 1]])
        #self._b = np.array([-0.5])
        
    def neuron(self, X):
        N = X.shape[1]
        net = np.zeros(N)
        threshold = 2
    
        for n in range(0,N):
            for j in range(0,self._dIn): # input neurons
                 net[n] += self._W[0,j] * X[j, n]
            net[n] += self._b * 1
        return net > threshold
    
    def DeltaTrain(self,X,T,eta,maxIter,maxErrorRate):
        N = X.shape[1]
        for i in range(maxIter):
            Y = self.neuron(X)
            err = ErrorRate(Y,T)
            print(err)
            if err < maxErrorRate:
                break
            deltaWkj = np.zeros(3)
            for j in range(2): # iterate over weights
                summe = 0
                for n in range(N): # iterate over features
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
                
def ErrorRate(Y,T):
    if Y.ndim==1 or Y.shape[0]==1:
        errors=Y!=T
        return errors.sum()/Y.size
    else: # für mehrere Ausgaben in one-hot Kodierung:
        errors=Y.argmax(0)!=T.argmax(0)
        return errors.sum()/Y.shape[1]        


iris_data = np.loadtxt('/path/to/iris.csv', delimiter=',')
X = iris_data[:100,:4].T
T = iris_data[:100,4].T

m=SLN(2,1)

m.DeltaTrain(X[:2],T,0.01,1000,0.02)
nnwplot.plotTwoFeatures(X[:2],T,m.neuron)
