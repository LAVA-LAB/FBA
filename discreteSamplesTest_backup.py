#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:38:59 2021

@author: thom
"""

import numpy as np

from core.mainFunctions import loadScenarioTable
from math import comb

from scipy.stats import beta as betaF

class Die():
    def __init__(self, events=np.arange(1,7), probabilities=np.ones(6)/6):
        if np.round(sum(probabilities), decimals=6) != 1:
            return
        elif len(events) != len(probabilities):
            return
        
        # Create probability bins
        self.events = events
        self.bins = [sum(probabilities[:i+1]) for i in range(len(events))]
        
    def throw(self):
        # Create random variable
        rand = sum(probabilities) * np.random.rand()
        
        # Check in which bin the random variable is
        idxs = [i for i, x in enumerate(self.bins) if rand < x]
        idx = int(idxs[0])
        value = self.events[idx]
        
        return idx, value

def determineProbabilities(scores, memory, d, beta):
    
    # Initialize probabilitie bounds
    p_low = np.zeros(len(scores))
    p_upp = np.zeros(len(scores))
    
    # Determine probability bounds for all events
    for i, score in enumerate(scores):
        
        k = throws - score + d
        
        key_lb = tuple( [throws, k,   beta] )
        key_ub = tuple( [throws, k-1, beta] )
        
        if k > throws:
            p_low[i] = 0                
        else:
            p_low[i] = 1 - memory[key_lb][1]
        p_upp[i] = 1 - memory[key_ub][0]    
        
        # k2 = throws - score
        
        # epsilon = k2 / throws + (np.sqrt(k2)/throws + (np.sqrt(k2)+1)/throws * 
        #                          ((d-1)*np.log(k2+d-1) + (d-1)/np.sqrt(k2) + 
        #                           np.log(1/beta)) )
        
        # prob_low2 = 1-epsilon
        
    return (p_low, p_upp)

#####


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


#####

throws = 25
beta = 0.01

tableFilename = 'input/probabilityTable_N='+str(throws)+ \
    '_beta='+str(beta)+'.csv'

memory = loadScenarioTable(tableFilename)

# Die settings (bias)
events = np.arange(1,7)
probabilities = np.array([0.1, 0.1, 0.2, 0.3, 0.25, 0.05])
# probabilities = np.array([1/2, 1/2])

# Number of decision variables is always 1
d = 1

# Create die
# die = Die(events, probabilities)
die = Die(events,probabilities)

ITERS = 10000
successes = np.zeros(len(events))
successes_half = 0

successesB = np.zeros(len(events))

scores = np.zeros((ITERS,len(events)))
values = np.zeros((ITERS,throws))

p_low  = np.zeros((ITERS,len(events)))
p_upp  = np.zeros((ITERS,len(events)))

p_lowB  = np.zeros((ITERS,len(events)))
p_uppB  = np.zeros((ITERS,len(events)))

def computeBetaPPF(N, k, d, beta):
    
    prob = betaF.ppf(beta, k+d, N-(d+k)+1)
    
    return prob

# For every iteration...
for it in range(ITERS):

    if it % 100 == 0:
        print('Trial number',it)    

    # Initialize scores at zero
    
    # Throw die N times
    for i in range(throws):
        # Throw die
        idx, value = die.throw()
        
        # Add score to the outcome
        scores[it,idx] += 1
        values[it,i] = value
    
    # Check if the actual probability is contained in the interval
    p_low[it], p_upp[it] = determineProbabilities(scores[it], memory, d, beta)
    
    ####
    
    for idx in range(len(events)):
        k = int(throws - scores[it,idx])
        
        if k != throws:
            # epsilon_upp = computeEpsilon(throws, beta, d, k)
            epsilon_upp = computeBetaPPF(throws, k, d, 1-beta/throws)
        else:
            epsilon_upp = 1
            
        if k != 0:
            # epsilon_low = computeEpsilon(throws, 1-beta, d, k-1)
            epsilon_low = computeBetaPPF(throws, k-1, d, beta/throws)
        else:
            epsilon_low = 0
            
        p_lowB[it,idx] = 1-epsilon_upp
        p_uppB[it,idx] = 1-epsilon_low
    
    ####
    
    for i, p in enumerate(probabilities):
        if p_low[it,i] <= p < p_upp[it,i]:
            successes[i] += 1
            
        if p_lowB[it,i] <= p < p_uppB[it,i]:
            successesB[i] += 1
    
    # print(' -- ',sum(scores[it, 3:]))
    
    p_low_b, p_upp_b = determineProbabilities([sum(scores[it, 3:])], memory, d, beta)
    
    if p_low_b <= sum(probabilities[3:]) < p_upp_b:
        successes_half += 1
            
# Compute overall fraction of successes
success_fraction = successes / ITERS
success_fraction_half = successes_half

success_fractionB = successesB / ITERS

print('Fractions of iters. where true probability was contained in interval:')
print('Old method:',success_fraction)
print('New method:',success_fractionB)

print('\nFractions of iters where probability to throw 4,5,6 was correctly bounded:')
print(success_fraction_half)

print('Average value thrown is',np.mean(values))

# %%

# Plot the distribution plus probability intervals
p_low_mean  = np.mean(p_low, axis=0)
p_low_stdev = np.std(p_low, axis=0)
p_upp_mean  = np.mean(p_upp, axis=0)
p_upp_stdev = np.std(p_upp, axis=0)

p_low_mean_b  = np.mean(p_low_b, axis=0)
p_low_stdev_b = np.std(p_low_b, axis=0)
p_upp_mean_b  = np.mean(p_upp_b, axis=0)
p_upp_stdev_b = np.std(p_upp_b, axis=0)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = [2,3,5]
freq = [0.3,0.2,0.5]
df = pd.DataFrame({'val.': events,'freq.': probabilities})
df.set_index('val.')['freq.'].plot.bar(rot=0, fill=False, label='True probability')

# Plot risk-complexity probability intervals
sns.pointplot(data=pd.DataFrame(p_low), dodge=True, join=False, ci='sd',
              color='red')

sns.pointplot(data=pd.DataFrame(p_upp), dodge=True, join=False, ci='sd',
              color='red')

# Plot traditional scenario approach probability intervals
sns.pointplot(data=pd.DataFrame(p_lowB), dodge=True, join=False, ci='sd',
              color='blue')

sns.pointplot(data=pd.DataFrame(p_uppB), dodge=True, join=False, ci='sd',
              color='blue')

ax.set_xlabel("Outcome")
ax.set_ylabel("probability")

# ax.legend(["True probability"])

ax.legend()
ax.set_title('Bounding probability distribution with N='+str(throws)+' sample throws')