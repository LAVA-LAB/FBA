#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 07:49:34 2021

@author: thom
"""

import numpy as np

def createTurbulenceSamples(N, length, dim):

    from core.UAV.dryden import DrydenGustModel
            
    # V_a = speed in 
    turb = DrydenGustModel(dt=1, b=5, h=20, V_a = 25, intensity="moderate")
    
    iters = N
    sample_length = length
    
    samples = np.zeros((iters,dim))
    
    for i in range(iters):
        
        if i % 100 == 0:
            print(' -- Create turbulence noise sample:',i)
        
        turb.reset()
        turb.simulate(sample_length)
        timeseries = turb.vel_lin
        
        samples[i,:] = timeseries[:,-1]
          
    return samples
        
# Main settings
N = 1000
length = 1000
dim = 3

# Create samples
samples = createTurbulenceSamples(N=N, length=length, dim=dim)

# Store samples
store_folder = "input/"
store_file   = "TurbulenceNoise_N="+str(N)+"_dim="+str(dim)

# Save file
np.savetxt(store_folder+store_file, samples, delimiter=",")