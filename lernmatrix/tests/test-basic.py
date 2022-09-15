# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:26 2022

@author: https://github.com/erickorsi

A few simple testsof base Lernmatrix.
"""
from lernmatrix import Lernmatrix

# Binary inputs
lm = Lernmatrix(4,3, binary=True)
lm.learn( X = [1,1,0,0], Y = [1,0,0] )
lm.learn( X = [0,1,0,1], Y = [0,1,0] )
lm.learn( X = [0,0,0,1], Y = [0,0,1] )
#lm.M
lm.recall( X = [0,1,0,1] )

# Errors
lm.learn( X = [1,1,0,0,1], Y = [1,0,0] )
lm.learn( X = [1,1,0,0], Y = [1,0,0,1] )
lm.learn( X = [1,1,0,0], Y = [1,0,2] )
lm.learn( X = 1, Y = [1,0,0] )

# Real-valued inputs
real_lm = Lernmatrix(5,3) # epsilon=0.001 may be better
real_lm.learn( X = [5.3,7.2,1,-4,1.2], Y = [1,0,0] )
real_lm.learn( X = [3.1,-3.2,0,-5,2], Y = [0,1,0] )
real_lm.learn( X = [4.2,-6.4,-1.3,2.1,0], Y = [0,0,1] )
#M = real_lm.M
real_lm.recall( X = [5.3,7.2,1,-4,1.2] )


# Binary inputs with auto-association
lm = Lernmatrix(6,6, binary=True, autoassociate=True)
lm.learn( X = [1,1,0,0,1,1], Y = [1,1,0,0,0,0] )
lm.learn( X = [0,1,0,1,0,1], Y = [0,0,1,1,0,0] )
lm.learn( X = [0,0,0,1,1,1], Y = [0,0,0,0,1,1] )
#lm.M
lm.recall( X = [1,1,0,0,1,1] )
lm.recall( X = [0,0,0,0,1,1] )
lm.recall( X = [1,0,0,0,1,1] )
lm.recall( X = [0,1,1,1,0,0] )


# Real-valued inputs with auto-association
real_lm = Lernmatrix(6,6, autoassociate=True) # epsilon=0.001 may be better
real_lm.learn( X = [5.3,7.2,1,-4,1.2,0], Y = [1,1,0,0,0,0] )
real_lm.learn( X = [3.1,-3.2,0,-5,2,-4], Y = [0,0,1,1,0,0] )
real_lm.learn( X = [4.2,-6.4,-1.3,2.1,0,1.1], Y = [0,0,0,0,1,1] )
#M = real_lm.M
real_lm.recall( X = [5.3,7.2,1,-4,1.2,0] )
real_lm.recall( X = [5,9,1.7,-5.1,3.1,1] )