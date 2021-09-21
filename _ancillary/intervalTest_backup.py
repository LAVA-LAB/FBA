#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:24:27 2021

@author: thom
"""

import numpy as np
from math import comb

from scipy.stats import beta as betaF

def createUniformSamples(N, low=-1, upp=1):
    
    if N > 1:
        rands = low + (upp-low)*np.random.rand(N)
    else:
        rands = low + (upp-low)*np.random.rand()
    
    return rands

def computeEpsilon(N, beta=0.1, d=1, k=0):
    
    eps_low = 1e-4
    eps_upp = 1 - 1e-4
    
    tol = 1e-9
    y_avg = 2*tol
    
    while np.abs(y_avg) > tol:
        eps_avg = (eps_upp + eps_low) / 2
        
        y_low = computeRoot(eps_low, N, beta, d, k)
        y_upp = computeRoot(eps_upp, N, beta, d, k)
        y_avg = computeRoot(eps_avg, N, beta, d, k)
                
        if (y_low > 0 and y_upp > 0):
            print('Error: both low/upp are above zero')
            return
        
        elif (y_low < 0 and y_upp < 0):
            print('Error: both low/upp are below zero')
            return
        
        else:
            if (y_avg >= 0 and y_low < 0) or (y_avg <= 0 and y_low > 0):
                eps_upp = eps_avg
            else:
                eps_low = eps_avg
                
    return eps_avg
    
def computeRoot(eps, N, beta=0.1, d=1, k=0):
    
    term1 = comb(k+d-1, k)
    term2 = sum([comb(N,i) * eps**(i) * (1-eps)**(N-i) for i in range(k+d)])
    
    return term1 * term2 - beta

def computeProbBeta(N, k, d, beta):
    
    prob = betaF.ppf(beta, k+d, N-(d+k)+1)
    
    return prob

def trial(N, beta, d, fixedWidth = 0):

    # Create interval for N samples
    samples = createUniformSamples(N)
    samples_abs = np.abs(samples)
    
    # if k > 0:
    #     samples_abs = np.sort(samples_abs)[0:-k]
    
    if fixedWidth != 0:
        samples_abs = samples_abs[samples_abs <= fixedWidth]
        k = N-len(samples_abs)
        #beta = beta/(k+1)
        beta = beta
    else:
        k = 0
        
    # print('Samples:',samples_abs,' - length:',len(samples_abs))
    # print('Discarded:',k)
    
    if k == N:
        return trial(N, beta, d, fixedWidth)
    
    '''
    if k != N:
        epsilon_upp = computeEpsilon(N, beta, d, k)
    else:
        # If k == N, all samples are discarded, so conclude prob.low = 0
        # epsilon_upp = 1
        
        return trial(N, beta, d, fixedWidth)
        
    if k != 0:
        epsilon_low = computeEpsilon(N, 1-beta, d, k-1)
    else:
        # If k == 0, no samples are discarded, so conclude prob.upp = 1
        # epsilon_low = 0
        
        return trial(N, beta, d, fixedWidth)
        
    p_low = 1-epsilon_upp
    p_upp = 1-epsilon_low
    
    print(p_low,'-',p_upp)
    '''
    
    p_low = 1 - computeProbBeta(N, k, d, 1-beta)
    p_upp = 1 - computeProbBeta(N, k-1, d, beta)
    
    # Determine semi-width of interval
    interval = np.max(samples_abs)
    
    true_prob = interval
    
    # if k != 0:
    #     samples_abs_plusOne = np.sort(samples_abs)[:len(samples_abs)+2]
    #     true_prob_upp = np.max(samples_abs_plusOne)
    
    if true_prob > p_upp:
        violation_upp = True
    else:
        violation_upp = False
        
    if true_prob < p_low:
        violation_low = True
    else:
        violation_low = False
    
    return p_low, violation_low, p_upp, violation_upp
    
    
    # if bound == 'low':
    #     beta = 1-beta
    
    # if k == N:
    #     epsilon = 0
    #     return False, epsilon
    # else:
    #     # Determine semi-width of interval
    #     interval = np.max(samples_abs)
        
    #     # Determine violation probability for this interval
    #     epsilon = computeEpsilon(N, beta, d, k)
        
    #     true_violation = 1 - interval
    
    # # Test how many random samples are indeed contained in the interval
    # '''
    # tests = 10000
    # test_samples = createUniformSamples(tests)
    
    # contained = sum(np.abs(test_samples) <= interval)
    # empirical_violation = (tests - contained) / tests
    
    # if empirical_violation < epsilon:
    #     return True
    # else:
    #     return False
    # '''
    
    # if bound == 'upp':
    #     if true_violation > epsilon:
    #         return True, epsilon
    #     else:
    #         return False, epsilon
    # else:
    #     if true_violation <= epsilon:
    #         return True, epsilon
    #     else:
    #         return False, epsilon
    
####
beta = 1e-2
d = 1
true_prob = 0.5

N = 25

trials = 10000
fails_upp = 0
fails_low = 0

p_low = np.zeros(trials)
p_upp = np.zeros(trials)

for tr in range(trials):
    
    if tr % 100 == 0:
        print('Trial number',tr)
    
    p_low[tr], outcome_low, p_upp[tr], outcome_upp = \
        trial(N, beta, d, true_prob)
    
    if outcome_low:
        fails_low += 1
    
    if outcome_upp:
        fails_upp += 1    
    
empirical_beta_upp = fails_upp / trials
print('Empirical upp.bound confidence level is:',empirical_beta_upp)

empirical_beta_low = fails_low / trials
print('Empirical low.bound confidence level is:',empirical_beta_low)

# %%

# Plot the distribution plus probability intervals
p_low_mean  = np.mean(p_low, axis=0)
p_low_stdev = np.std(p_low, axis=0)
p_upp_mean  = np.mean(p_upp, axis=0)
p_upp_stdev = np.std(p_upp, axis=0)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = [2,3,5]
freq = [0.3,0.2,0.5]

plt.bar(x=0, height=true_prob, fill=False, label='True probability')

sns.pointplot(data=pd.DataFrame(p_low), dodge=True, join=False, ci='sd',
              color='red')

sns.pointplot(data=pd.DataFrame(p_upp), dodge=True, join=False, ci='sd',
              color='red')

ax.set_xlabel("Outcome")
ax.set_ylabel("probability")

# ax.legend(["True probability"])

ax.legend()
# ax.set_title('Bounding probability distribution with N='+str(throws)+' sample throws')