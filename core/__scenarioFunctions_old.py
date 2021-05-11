# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:38:52 2021

@author: Thom Badings
"""

import numpy as np
import copy

import math

from .commons import printWarning

def calculateTransitionProbabilities_SA(setup, abstr, baseDim, 
                                        action_id, mu, S):
    
    # Sample N scenarios using the defined distribution    
    samples = np.random.multivariate_normal(mu, S, size=setup['sa']['samples'])
    
    # Create empty rootfinding table for memory
    rootfinding_table = []
    
    # Factorial of N is equal for every iteration, so calculate up front
    log_factorial_N = setup['sa']['log_factorial_N']
    xInitLow = [1e-5, 1-1e-5]
    xInitUpp = [1e-5, 1-1e-5]
    threshold = 1e-2
    
    # Number of decision variables always equal to one
    d = 1
    k = setup['sa']['samples']
    
    # Initialize vectors for probability bounds
    probability_low = np.zeros(abstr['nr_regions'])
    probability_upp = np.zeros(abstr['nr_regions'])
    
    counts = np.zeros(abstr['nr_regions'])
    for s in samples:
        center = ((s+1) // 2) * 2
        key = tuple(center)
        if key in abstr['targetPointsInv']:
            counts[abstr['targetPointsInv'][ tuple(center) ]] += 1

    nonEmpty = counts > 0
    
    # Count number of samples not in any region (i.e. in absorbing state)
    k = sum(counts)
    key = tuple( [setup['sa']['samples'], d, k, setup['sa']['confidence']] )
    keyInMemory = key in abstr['memory']
    
    if not keyInMemory:    
        rootfinding_table, zeroEpsilon_low = calculateEpsilon(rootfinding_table, xInitLow, setup['sa']['samples'], log_factorial_N, d, k, setup['sa']['confidence'], threshold)
        rootfinding_table, zeroEpsilon_upp = calculateEpsilon(rootfinding_table, xInitUpp, setup['sa']['samples'], log_factorial_N, d, k-1, 1-setup['sa']['confidence'], threshold)

        deadlock_low = 1 - zeroEpsilon_low
        deadlock_upp = 1 - zeroEpsilon_upp
        
    else:
        deadlock_low = abstr['memory'][key][0]
        deadlock_upp = abstr['memory'][key][1]

    # Enumerate over all the non-zero bins
    for i, count in zip(np.arange(abstr['nr_regions'])[nonEmpty], counts[nonEmpty]):
        
        # Only do something if the count is larger than zero
        if count > 0:
            # Determine number of discarded samples
            k = setup['sa']['samples'] - count
            
            # Determine lower bound on probability
            key = tuple( [setup['sa']['samples'], d, k, setup['sa']['confidence']] )
            keyInMemory = key in abstr['memory']
            
            if not keyInMemory:
                print('-----',setup['sa']['samples'],d,k,setup['sa']['confidence'],'does not exist, so create')
                
                rootfinding_table, epsilon_low = calculateEpsilon(rootfinding_table, xInitLow, setup['sa']['samples'], log_factorial_N, d, k, setup['sa']['confidence'], threshold)
                rootfinding_table, epsilon_upp = calculateEpsilon(rootfinding_table, xInitUpp, setup['sa']['samples'], log_factorial_N, d, k-1, 1-setup['sa']['confidence'], threshold)
                
                probability_low[i] = 1 - epsilon_low
                probability_upp[i] = 1 - epsilon_upp
                
                abstr['memory'][key] = [probability_low[i], probability_upp[i]]
                
            else:
                probability_low[i] = abstr['memory'][key][0]
                probability_upp[i] = abstr['memory'][key][1]
    
    returnDict = {
        'lb': probability_low,
        'ub': probability_upp,
        'deadlock_lb': deadlock_low,
        'deadlock_ub': deadlock_upp,
        'approx': probability_low
    }
    
    return abstr['memory'], returnDict

def calculateEpsilon(table, xInit, N, log_factorial_N, d, k, beta, threshold):
        if k == N:
            # If no samples are in the hull at all, then the probability is 
            # zero a-priori (i.e. epsilon is 1)
            epsilon = 1
        elif k < 0:
            # If all "plus one" samples are in hull the 
            # probability is one a-priori (i.e. epsilon is 0)
            epsilon = 0
        else:
            # Term to account for discarded samples (alpha)
            alpha = 1#math.comb(k+d-1, k)
            
            # Solve the rootfinding problem to find the values of epsilon
            table, epsilon, root = root_finding(table, N, log_factorial_N, beta, k, d, alpha, xInit, maxq=100, threshold=threshold)
        
        # Return bounds in reversed order, since probability is 1-epsilon
        return table, epsilon
    
def root_finding(table, N, log_factorial_N, beta, k, d, alpha_term, xInit, maxq, threshold):
        # Solve root finding problem to find epsilon        
        x = [xInit[0], xInit[1], 0]*3
        y = [0,0,0]
        
        q = 0
        while q < maxq and (abs(x[0]-x[1]) > threshold or q == 0):
            q += 1
        
            # Determine new x-point
            x[2] = np.mean([x[0], x[1]])
        
            for i in [0,1,2]:
                # Initialize y at zero
                y[i] = 0
                
                # Upper limit on summation
                j_max = k+d
                
                # Search if this calculation was already performed

                # Find the highest value for k for which calculation was already performed
                # Extract partial table for calculations that are already performed
                partial_table = [row for row in table if row[0] == N and row[1] == x[i] and row[2]+1 <= j_max]
                
                key = tuple([ N, x[i] ])
                
                
                max_table_entry = partial_table
                
                # If any previous calculation was found
                if len(partial_table) > 0:
                    
                    # Sort the partial table to extract the highest value of k
                    partial_table.sort(key=lambda x:x[2], reverse=True)
                    
                    # Highest value is now stored in the first list item
                    idx = partial_table[0][2] + 1
                    y[i] += partial_table[0][3]

                else:
                    idx = 0
                    
                if idx < j_max:
                    # Parameters to convert epsilon to integer number
                    dec = int(6)
                    factor = int(10**dec)
                    
                    # Calculate as much as possible outside the loop over j
                    log_round_x = math.log( int(factor*round(x[i],dec)))
                    log_round_minX = math.log(int(factor*round(1-x[i],dec)))
                    log_factor = math.log(factor)
                
                for j in range(idx,j_max):
                    
                    # Logarithmic computation to avoid numerical overflows
                    numerator = log_factorial_N + \
                        j*log_round_x + \
                        (N-j)*log_round_minX
                    denominator = math.log(math.factorial(j)) + \
                        math.log(math.factorial(N-j)) + \
                        j*log_factor + \
                        (N-j)*log_factor
                    log_x = numerator - denominator

                    y[i] += np.exp(log_x)
                    
                # And insert new entry
                if idx < j_max:
                    table.append([ N, copy.deepcopy(x[i]), j_max-1, y[i] ])
                
                # Subtract beta and alpha term to obtain final result
                y[i] -= beta/alpha_term
                
            if y[0] == 0 and y[1] == 0:
                printWarning('Error: both y_a and y_c are equal to zero')
                break
            elif y[0] > 0 and y[1] > 0:
                printWarning('Error: both y_a and y_c are above zero')
                break
            elif y[0] < 0 and y[1] < 0:
                printWarning('Error: both y_a and y_c are below zero')
                break
            else:
                if (y[0] < 0 and y[2] < 0) or (y[0] > 0 and y[2] > 0):
                    x[0] = x[2]
                else:
                    x[1] = x[2]
                    
        return table, x[2], y[2]
    
