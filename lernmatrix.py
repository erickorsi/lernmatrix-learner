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

        return 0

    def recall(self, X):
        '''
        '''
        # Validation of data
        _input_validation(X, self.x_length)

        # Dot product of matrix with input
        Y_temp = np.asarray(np.dot(self.M, X)).reshape(-1)

        # Get binary array based on max value of result
        y_max = np.amax(Y_temp)
        Y = np.array([1 if y==y_max else 0 for y in Y_temp])

        return Y



'''
lm = Lernmatrix(4,3)
lm.learn( X = [1,1,0,0], Y = [1,0,0] )
lm.learn( X = [0,1,0,1], Y = [0,1,0] )
lm.learn( X = [0,0,0,1], Y = [0,0,1] )

lm.recall( X = [0,1,0,1] )
'''


def _input_validation(input, length):
        '''
        Validates the input sequence.
        Must be of the same length as the respective matrix size;
        Must be composed of only 0s and 1s.

        Parameters
        ----------
        input : array or list
            Input sequence to be validated (can be X or Y).
        length : int
            Expected length of this input according to the matrix structure.

        Returns
        ----------
        None.
        Raises Exception if invalid.
        '''
        try:
            if (len(input)!=length): # Size of inputs
                raise ValueError("Invalid length. Sequence has size {} but matrix requires size {}".format(len(input), length))
            if (np.any([(i!=0 and i!=1) for i in input])): # Values of inputs
                raise ValueError("Invalid values. Sequence must be binary (composed of 0s and 1s).")
        except TypeError:
            raise TypeError("Invalid input type. Input must be an itterable list or array.")