# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:06 2022

@author: https://github.com/erickorsi

Lernmatrix class methods.
"""
import numpy as np

class Lernmatrix():
    '''
    '''

    def __init__(self, x_length, y_length, epsilon=1):
        self.x_length = x_length
        self.y_length = y_length
        self.epsilon = epsilon

        # Creates the initial instance of the matrix filled with 0s
        self.M = np.matrix(np.tile(np.zeros(x_length),(y_length,1)))

    def learn(self, X, Y):
        '''
        '''
        # Validation of data
        if (len(X)!=self.x_length or len(Y)!=self.y_length): # Size of inputs
            return 1
        if (np.any([(x!=0 and x!=1) for x in X]) or np.any([(y!=0 and y!=1) for y in Y])): # Values of inputs
            return 2

        # Runs inputs through Lernmatrix ruleset
        for row in range(self.y_length):
            for col in range(self.x_length):
                if (Y[row]==0):
                    val = 0
                elif (X[col]==1):
                    val = self.epsilon
                else:
                    val = -self.epsilon

                # Changes values in the matrix
                self.M[row,col] += val

        return 0

    def recall(self, X):
        '''
        '''
        # Validation of data
        if (len(X)!=self.x_length): # Size of input
            return 1
        if (np.any([(x!=0 and x!=1) for x in X])): # Values of input
            return 2

        # Dot product of matrix with input
        Y_temp = np.asarray(np.dot(self.M, X)).reshape(-1)

        # Get binary array based on max value of result
        y_max = np.amax(Y_temp)
        Y = np.array([1 if y==y_max else 0 for y in Y_temp])

        return Y




'''
lm = Lernmatrix(3,4)
lm.M
lm.learn([1,1,1],[1,1,1,1])
lm.M
lm.learn([1,0,0],[0,1,1,0])
lm.M

M = lm.M
x = [1,0,0]
Y_temp = np.asarray(np.dot(M, x)).reshape(-1)
'''
