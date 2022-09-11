# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:06 2022

@author: https://github.com/erickorsi

Lernmatrix class methods.
"""
import numpy as np

class Lernmatrix():

    def __init__(self, x_length, y_length, epsilon=1):
        '''
        '''
        self.x_length = x_length
        self.y_length = y_length
        self.epsilon = epsilon

        # Creates the initial instance of the matrix filled with 0s
        self.M = np.matrix(np.tile(np.zeros(x_length),(y_length,1)))

    def learn(self, X, Y):
        '''
        '''
        # Validation of data
        # check size and binary values

        for row in range(self.y_length):
            for col in range(self.x_length):
                if (Y[row] == 0):
                    val = 0
                elif (X[col] == 1):
                    val = self.epsilon
                else:
                    val = -self.epsilon

                self.M[row,col] += val

    def recall():
        '''
        '''



lm = Lernmatrix(3,4)
lm.learn([1,1,1],[1,1,1,1])
lm.M
lm.learn([1,0,0],[0,1,1,0])
lm.M