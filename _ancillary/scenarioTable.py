#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import beta as betaF

def computeBetaPPF(N, k, d, beta):
    
    prob = betaF.ppf(beta, k+d, N-(d+k)+1)
    
    return prob

def createTable(N, beta):
    
    show_every = 100
    
    beta_bar = beta / np.ceil(N/2)
    
    epsL = np.zeros(N+1)
    epsU = np.zeros(N+1)
    
    for k in range(N+1):
        if k % show_every == 0:
            print('Compute for N =',N,'and k =',k)
        
        if k == N:
            
            epsL[k] = computeBetaPPF(N=N, k=k-1, d=1, beta=beta_bar)
            epsU[k] = 1
        
        else:
        
            epsL[k] = computeBetaPPF(N=N, k=k, d=1, beta=beta_bar)
            epsU[k] = computeBetaPPF(N=N, k=k, d=1, beta=1-beta_bar)
        
    store_folder = "input/"
    filename = "probabilityTable_new_N="+str(N)+"_beta="+str(beta)+".csv"
    
    N_list = np.full(N+1, N, dtype=int)
    k_list = np.arange(0,N+1)
    beta_list = beta * np.ones(N+1)
    
    contents = np.vstack((N_list, k_list, beta_list, epsL, epsU)).T
    contents_df = pd.DataFrame(data=contents, dtype=np.array([int, int, float, float, float]))
    
    contents_df.to_csv(store_folder+filename, index=False, header=False)
    
    # np.savetxt(store_folder+filename, contents, delimiter=",")

#########    

Nlist = np.array([25*2**n for n in range(0,12)])
beta = 0.1

for N in Nlist:
    
    print('\nTABULE FOR N =',N,'\n')
    
    createTable(N, beta)