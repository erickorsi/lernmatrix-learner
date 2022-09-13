# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:06 2022

@author: https://github.com/erickorsi

Lernmatrix class methods.
"""
import numpy as np

from .utils import _input_validation

class Lernmatrix():
    '''
    Steinbuch Lernmatrix object.
    The base Lernmatrix works with binary inputs.
    Using a modified Lernmatrix ruleset, it is possible to accept real-valued inputs
    (in this case, the expected classes should still be binary sequences).

    Parameters
    ----------
    x_length : int
        Size of main input list and number of columns in the lernmatrix.
    y_length : int
        Size of output list and number of rows in the lernmatrix.
    epsilon : float, default=1.0
        Increment value of the lernmatrix learning process.
        Can be any positive number.

    Attributes
    ----------
    x_length : int
        Size of main input list and number of columns in the lernmatrix.
    y_length : int
        Size of output list and number of rows in the lernmatrix.
    epsilon : float, default=1.0
        Increment value of the lernmatrix learning process.
        Can be any positive number.
    M : numpy matrix
        The lernmatrix itself, initiallized according to x_length and y_length.
        All initial values are 0.

    Methods
    ----------
    learn(...)
    recall(...)

    Information on the methods can be seen with help() function.
    '''

    def __init__(self, x_length, y_length, epsilon=1.0):
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
            Sequence representing the main input.
            Must be the same length as the Lernmatrix input length.
            If binary, the base Lernmatrix will be used.
            If real-valued, a modified Lernmatrix will be used.
        Y : array or list
            Binary sequence representing the expected output (associated input).
            Must be the same length as the Lernmatrix output length.
            Ex.: [1,0,0] or [0,1,0]
            Represents the expected class.

        Returns
        ----------
        None
        '''
        # Validation of data
        _ =_input_validation(Y, self.y_length, binary=True)
        status =_input_validation(X, self.x_length)


        # Runs inputs through Lernmatrix ruleset
        for row in range(self.y_length):
            for col in range(self.x_length):

                # Regular Lermatrix
                if status == 0:
                    if (Y[row]==0):
                        val = 0
                    elif (X[col]==1):
                        val = self.epsilon
                    else:
                        val = -self.epsilon

                # Real-valued Lernmatrix
                elif status == 1:
                    if (Y[row]==0):
                        val = 0
                    elif (X[col]==0):
                        val = self.epsilon
                    else:
                        val = X[col]

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
        status = _input_validation(X, self.x_length)

        # Regular Lermatrix
        if status == 0:
            # Dot product of matrix with input
            Y_temp = np.asarray(np.dot(self.M, X)).reshape(-1)
            # Get binary array based on max value of result
            y_max = np.amax(Y_temp)
            Y = np.array([1 if y==y_max else 0 for y in Y_temp])

        # Real-valued Lernmatrix
        elif status == 1:
            # Get multiplicative matrix from inverse input
            X_inv = np.array([1/x for x in X])
            M_temp = np.asarray(self.M) * X_inv
            # Get sum of rows from absolute asymptotic matrix
            Y_temp = np.sum(np.abs(np.tanh(M_temp-1)), axis=1)
            # Get binary array based  on min value of result
            y_min = np.amin(Y_temp)
            Y = np.array([1 if y==y_min else 0 for y in Y_temp])

        return Y


