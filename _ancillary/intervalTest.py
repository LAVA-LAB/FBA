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

def computeBetaPPF(N, k, d, beta):
    
    prob = betaF.ppf(beta, k+d, N-(d+k)+1)
    
    return prob

def trial(N, beta, d, fixedWidth = 0):

    # Create interval for N samples
    samples = createUniformSamples(N)
    samples_abs = np.abs(samples)
    
    if fixedWidth != 0:
        samples_abs = samples_abs[samples_abs <= fixedWidth]
        k = N-len(samples_abs)
        beta = beta
    else:
        k = 10
        samples_abs = np.sort(samples_abs)[0:-k]
    
    # if k == N:
    #     return trial(N, beta, d, fixedWidth)
    
    beta_bar = beta/np.ceil(N/2)
        
    if k == N:
        V_low = computeBetaPPF(N=N, k=k-1, d=1, beta=beta_bar)
        V_upp = 1
        
    else:
        V_upp = computeBetaPPF(N, k, d, 1-beta_bar)
    
        if k != 0:
            V_low = computeBetaPPF(N, k-1, d, beta_bar)
        else:
            V_low = 0
    
    # Determine semi-width of interval
    if k != N:
        interval = np.max(samples_abs)
        
    else:
        interval = 0
        
    V_true = 1 - interval
        
        
    if V_true > V_upp:
        violation_upp = True
    else:
        violation_upp = False
        
    if V_true < V_low:
        violation_low = True
    else:
        violation_low = False
    
    return V_low, violation_low, V_upp, violation_upp
    
####
beta = 0.1
d = 1
true_prob = 0.3

N = 1000

trials = 1000
fails_upp = 0
fails_low = 0

p_low = np.zeros(trials)
p_upp = np.zeros(trials)

for tr in range(trials):
    
    if tr % 100 == 0:
        print('Trial number',tr)
    
    V_low, violation_low, V_upp, violation_upp = \
        trial(N, beta, d, true_prob)
    
    p_low[tr] = 1 - V_upp
    p_upp[tr] = 1 - V_low
    
    if violation_low:
        fails_low += 1
    
    if violation_upp:
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