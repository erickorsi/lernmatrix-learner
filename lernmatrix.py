# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:06 2022

@author: https://github.com/erickorsi

Lernmatrix class methods.
"""
import numpy as np

class Lernmatrix():

    def __init__(self, input_length, output_length):
        '''
        '''
        self.input_length = input_length
        self.output_length = output_length

        # Creates the initial instance of the matrix filled with 0s
        self.M = np.matrix(np.tile(np.zeros(input_length),(output_length,1)))

    def learn():
        '''
        '''
        

    def recall():
        '''
        '''