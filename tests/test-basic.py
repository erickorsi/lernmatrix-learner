# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:26 2022

@author: https://github.com/erickorsi

A few simple testsof base Lernmatrix.
"""
from ..lernmatrix import Lernmatrix

lm = Lernmatrix(4,3)
lm.learn( X = [1,1,0,0], Y = [1,0,0] )
lm.learn( X = [0,1,0,1], Y = [0,1,0] )
lm.learn( X = [0,0,0,1], Y = [0,0,1] )

lm.recall( X = [0,1,0,1] )
