#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:15:52 2021

@author: thom
"""
#!/usr/bin/env python3

# Load general packages
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
import matplotlib.pyplot as plt # Import Pyplot to generate plots

# Load main classes and methods
from core.commons import createDirectory, tocDiff
from core.mainFunctions import steadystateCovariance, steadystateCovariance_sdp

from core.filterBasedAbstraction import MonteCarloSim

#-----------------------------------------------------------------------------
# Code below is repeated every iteration of the iterative scheme
#-----------------------------------------------------------------------------

def filterBasedScheme(Ab, case_id):

    # Save case-specific data in Excel
    output_file = Ab.setup.directories['outputFcase'] + \
        Ab.setup.time['datetime'] + '_data_export.xlsx'
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')    

    # Only perform code below is a new run is chosen
    if Ab.setup.main['newRun'] is True:
    
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        print('START FILTER-BASED ABSTRACTION') 
        
        # Create folder to save results
        createDirectory( Ab.setup.directories['outputFcase'] )    
        
        # Compute steady state covariance matrices (best/worst-case)
        k_stst = Ab.setup.mdp['k_steady_state']
        if k_stst != None:
              
            if Ab.setup.main['covarianceMode'] == 'SDP':
                func = steadystateCovariance_sdp
            else:
                func = steadystateCovariance
            
            Ab.km[1]['steady'] = func(
                [ Ab.km[1][k]['cov_tilde'] 
                  for k in range(k_stst, len(Ab.km[1])) ], 
                                          verbose=False)
        
        # Calculate transition probabilities
        Ab.defTransitions()
        
        # Build MDP
        model_size = Ab.buildMDP()
        
        # Write model size results to Excel
        model_size_df = pd.DataFrame(model_size, index=[case_id])
        model_size_df.to_excel(writer, sheet_name='Model size')
        
        # Solve the MDP
        Ab.solveMDP()
        
        # Load data into dataframes
        policy_df   = pd.DataFrame( np.stack(Ab.mdp.MAIN_DF['opt_action'].to_numpy()) )
        delta_df    = pd.DataFrame( np.stack(Ab.mdp.MAIN_DF['opt_delta'].to_numpy()) )
        reward_df   = pd.DataFrame( Ab.mdp.opt_reward )
        
        # Write dataframes to a different worksheet
        policy_df.to_excel(writer, sheet_name='Optimal policy')
        delta_df.to_excel(writer, sheet_name='Optimal delta')
        reward_df.to_excel(writer, sheet_name='Optimal reward')
    
    if Ab.setup.montecarlo['enabled']:
        # Perform monte carlo simulation for validation purposes
        
        mc_obj = MonteCarloSim(Ab, iterations = Ab.setup.montecarlo['iterations'],
                                   init_states = Ab.setup.montecarlo['init_states'] )
        
        Ab.mc = {'reachability_probability': mc_obj.results['reachability_probability'],
                 'traces': mc_obj.traces }
        
        # Store Monte Carlo results as dataframe
        MCsims_df = pd.DataFrame( Ab.mc['reachability_probability'], 
                                  index=Ab.abstr['P'].keys() )
        
        # Clean monte carlo object to save space
        del mc_obj
            
        # Write Monte Carlo results to Excel
        MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
    
    # Plot results
    Ab.generatePlots( delta_value = 1, 
                      max_delta = max(Ab.setup.all_deltas),
                      case_id = case_id,
                      writer = writer)
    
    # %%
    
    # Store run times of current iterations        
    time_df = pd.DataFrame( data=Ab.time, index=[case_id] )
    time_df.to_excel(writer, sheet_name='Run time')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    plt.close('all')
        
    return Ab
            
    
    
    
    
    