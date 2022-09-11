# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:06 2022

@author: https://github.com/erickorsi

Lernmatrix class methods.
"""
import numpy as np

from ..utils import _input_validation

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
        Learning process of Lernmatrix of a single example.
        Follows the set of rules described in the learning phase.

        Parameters
        ----------
        X : array or list
            Binary sequence representing the main input.
            Must be the same length as the Lernmatrix input length.
            Ex.: [1,0,0,1,...,1,0,1]
        Y : array or list
            Binary sequence representing the expected output (associated input).
            Must be the same length as the Lernmatrix output length.
            Ex.: [1,0,0,1,...,1,0,1]

        Returns
        ----------
        None
        '''
        # Validation of data
        _input_validation(X, self.x_length)
        _input_validation(Y, self.y_length)

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

    def recall(self, X):
        '''
        Recall process of Lernmatrix of a single example.
        Follows the set of rules described in the recall phase.

        Parameters
        ----------
        X : array or list
            Binary sequence representing the main input.
            Must be the same length as the Lernmatrix input length.
            Ex.: [1,0,0,1,...,1,0,1]

        Returns
        ----------
        Y : array
            Binary sequence representing the calculated output.
        '''
        # Validation of data
        _input_validation(X, self.x_length)

        # Dot product of matrix with input
        Y_temp = np.asarray(np.dot(self.M, X)).reshape(-1)

        # Get binary array based on max value of result
        y_max = np.amax(Y_temp)
        Y = np.array([1 if y==y_max else 0 for y in Y_temp])

        return Y
