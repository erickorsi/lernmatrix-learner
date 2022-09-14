# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:19 2022

@author: https://github.com/erickorsi
"""
import numpy as np

def _input_validation(input, length, binary=False):
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
        binary : boolean, default=False
            If the sequence needs to be binary or not.
            For the output, the sequence should be binary, defining different classes in a simple manner.

        Returns
        ----------
        None
        Raises Exception if invalid.
        '''
        try:
            if (len(input)!=length): # Input size
                raise ValueError("Invalid length. Sequence has size {} but matrix requires size {}".format(len(input), length))
            if (np.any([(i!=0 and i!=1) for i in input])): # Input values
                if (binary==True):
                    raise ValueError("Invalid values. If binary=True, the values sequences must be composed of 0s and 1s. The classes should always be binary.")
                return 1 # Non-binary values. Will be treated as real-valued input.
            else:
                if (binary==True):
                    return 0 # Binary valued classification.
                else:
                    return 1 # Real-valued classification may still have binary sequences.
        except TypeError: # Input datatype
            raise TypeError("Invalid type. Input must be a non-string itterable list or array.")