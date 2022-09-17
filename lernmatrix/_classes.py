# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:06 2022

@author: https://github.com/erickorsi

Lernmatrix class methods.
"""
import numpy as np
from random import sample

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
    binary : boolean, default=False
        Indicates if the inputs are expected to be binary or real-valued.
        Real-valued inputs have different learning and recall processes.
        By default, the model uses the modified Lernmatrix for real values.
    autoassociate : boolean, default=False
        Indicates if the Lernmatrix will auto-associate expected output.
        Simplified simulation of auto-association of the human hippocampus,
        where the expected output is re-inserted as an input, but with slight changes,
        in order to associate the expected output with slightly different sequences.
        Allows the model to reduce error from noisy inputs.
        Can only be used for cases where classes are not defined by single values in the sequence. Ex.:
        YES: [1,1,0,0,0,0], [0,0,1,1,0,0] and [0,0,0,0,1,1]
        NO : [1,0,0], [0,1,0] and [0,0,1]
        Can only be used With square matrices, where length of X = length of Y.
    bit_error : float, default=0.01
        Only used when autoassociate = True.
        The percentage of values changed from the output when auto-associating,
        rounded up.

    Attributes
    ----------
    x_length : int
        Call the size of main input list and number of columns in the lernmatrix.
    y_length : int
        Call the size of output list and number of rows in the lernmatrix.
    epsilon : float, default=1.0
        Call the epsilon value of the model.
    binary : boolean, default=False
        Call the binary flag of the model.
    autoassociate : boolean, default=False
        Call the autoassociate flag of the model.
    bit_error : float, default=0.01
        Call the bit_error value of the object.
        Only used when autoassociate = True.
    M : numpy matrix
        The lernmatrix itself, initiallized according to x_length and y_length.
        All initial values are 0.

    Methods
    ----------
    learn(...)
    recall(...)

    Information on the methods can be seen with help() function.
    '''

    def __init__(self, x_length, y_length, epsilon=1.0, binary=False, autoassociate=False, bit_error=0.01):
        self.x_length = x_length
        self.y_length = y_length
        self.epsilon = epsilon
        self.binary = binary
        self.autoassociate = autoassociate
        self.bit_error = bit_error

        # Creates the initial instance of the matrix filled with 0s
        self.M = np.matrix(np.tile(np.zeros(x_length),(y_length,1)))

    def learn(self, X, Y, *args):
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
        None.
        '''
        # Args handling
        if (len(args)==0):
            recurring = False
        else:
            recurring = args[0]

        # Validation of data
        _input_validation(Y, self.y_length, binary=True)
        _input_validation(X, self.x_length)

        # Runs inputs through Lernmatrix ruleset
        for row in range(self.y_length):
            for col in range(self.x_length):

                if (Y[row]==0):
                    val = 0 # y=0 adds 0 to matrix cell, regardless of x.
                elif (X[col]==0):
                    if (self.binary==True):
                        val = -self.epsilon # y=1 and x=0 adds -epsilon, in binary lernmatrix
                    else:
                        val = self.epsilon # y=1 and x=0 adds epsilon, in real-valued lernmatrix
                elif (self.binary==True):
                    val = self.epsilon # y=1 and x=1 adds epsilon, in binary lernmatrix
                else:
                    val = X[col] # y=1 and x=1 adds x, in real-valued lernmatrix

                # Changes values in the matrix
                self.M[row,col] += val

        # Auto-association
        if (self.autoassociate==True and recurring==False):
            vals_change = np.random.randint(0, len(Y), size=np.ceil(len(Y)*self.bit_error).astype(int)) # Random indexes to change
            Y_mod = np.copy(Y)
            for n in vals_change:
                Y_mod[n] = np.abs(Y_mod[n]-1) # Changes the value between 0 and 1
            # Auto-associate the changed expected output with the expected output
            self.learn(Y_mod, Y, "recur")

    def recall(self, X, *args):
        '''
        Recall process of Lernmatrix of a single example.
        Follows the set of rules described in the recall phase.

        Parameters
        ----------
        X : array or list
            Sequence representing the main input.
            Must be the same length as the Lernmatrix input length.
            If binary, the base Lernmatrix will be used.
            If real-valued, a modified Lernmatrix will be used.

        Returns
        ----------
        Y : array
            Binary sequence representing the calculated output.
        '''
        # Args handling
        if (len(args)==0):
            recurring = False
        else:
            recurring = args[0]

        # Validation of data
        _input_validation(X, self.x_length)

        # Binary Lermatrix
        if (self.binary==True):
            # Dot product of matrix with input
            Y_temp = np.asarray(np.dot(self.M, X)).reshape(-1)
            # Get binary array based on max value of result
            y_max = np.amax(Y_temp)
            Y = np.array([1 if y==y_max else 0 for y in Y_temp])

        # Real-valued Lernmatrix
        else:
            # Get multiplicative matrix from inverse input
            X_inv = np.array([1/x if x!=0 else 1/self.epsilon for x in X])
            M_temp = np.asarray(self.M) * X_inv
            # Get sum of rows from absolute asymptotic matrix
            Y_temp = np.sum(np.abs(np.tanh(M_temp-1)), axis=1)
            # Get binary array based on min value of result
            y_min = np.amin(Y_temp)
            Y = np.array([1 if y==y_min else 0 for y in Y_temp])
        
        # Auto-association validation
        if (self.autoassociate==True and recurring==False):
            # Attempts to get the class associated to the inaccurate result in case of classification error
            Y = self.recall(Y, "recur")

        return Y

    def fit(self, X, Y, binary=False, autoassociate=False, bit_Error=0.01):
        '''
        Learning process using a dataset.
        Runs the learn method for each element in the dataset.

        Parameters
        ----------
        X : pandas DataFrame
            Dataframe containing the dependent variables, or the sequences for input data.
        Y : pandas DataFrame
            Dataframe containing the classes for each element in X.
            Must be the same length and order as X.

        Returns
        ----------
        None.
        '''
 