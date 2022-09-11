# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:19 2022

@author: https://github.com/erickorsi
"""
import numpy as np

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
        None
        Raises Exception if invalid.
        '''
        try:
            if (len(input)!=length): # Input size
                raise ValueError("Invalid length. Sequence has size {} but matrix requires size {}".format(len(input), length))
            if (np.any([(i!=0 and i!=1) for i in input])): # Input values
                raise ValueError("Invalid values. Sequence must be binary (composed of 0s and 1s).")
        except TypeError: # Input datatype
            raise TypeError("Invalid type. Input must be an itterable list or array.")