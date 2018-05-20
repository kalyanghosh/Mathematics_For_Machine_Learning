# -*- coding: utf-8 -*-
"""
Created on Sat May 19 19:45:09 2018

@author: Kalyan
"""

#Full Singular Value Decomposition
import numpy as np

A=np.array([[1,1,1,0,0],
           [3,3,3,0,0],
           [4,4,4,0,0],
           [5,5,5,0,0],
           [0,2,0,4,4],
           [0,0,0,5,5],
           [0,1,0,2,2]])


np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

print ('------FULL SVD EXAMPLE--------')
U,E,VT=np.linalg.svd(A,full_matrices=True)

print ("U:\n {}".format(U))
print ("E:\n {}".format(E))
print ("VT:\n {}".format(VT))

#Full Singular Value Decomposition

print ('------TRUNCATED SVD EXAMPLE--------')
U,E,VT=np.linalg.svd(A,full_matrices=False)

print ("U:\n {}".format(U))
print ("E:\n {}".format(E))
print ("VT:\n {}".format(VT))