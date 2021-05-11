#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ..commons import nchoosek

def gears_order(s):
    '''
    Function to calculate gears discretization parameters, 
    based on the order `s` (input)
    ''' 
    beta0 = 0                   # Define beta0
    for i in range(1,s+1):
        beta0 = beta0 + 1/i     # Add to sum of beta0
    beta0 = beta0**(-1)
    
    alpha = np.zeros(s)
    alphaT = np.zeros(s)
    for i in range(1,s+1):
        alpha[i-1] = (-1)**(i+1)*beta0
        alphaT[i-1] = 0
        for j in range(i,s+1):
            alphaT[i-1] = alphaT[i-1] + j**(-1)*nchoosek(j,i)
        alpha[i-1] = alpha[i-1]*alphaT[i-1]
        
    alphaSum = 0
    for j in range(s):
        alphaSum += alpha[j]
        
    return alpha, beta0, alphaSum