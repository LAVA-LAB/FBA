#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:10:59 2021

@author: thom
"""

# Load general packages
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
import math                     # Import Math for mathematical operations
import matplotlib.pyplot as plt # Import Pyplot to generate plots

# Load main classes and methods
from core.preprocessing.user_interface import load_PRISM_result_file
from core.commons import createDirectory

#-----------------------------------------------------------------------------
# Code below is repeated every iteration of the iterative scheme
#-----------------------------------------------------------------------------

def iterativeScheme(Ab):

    # Initialize case ID at zero
    case_id = 0
    
    # Create empty DataFrames to store iterative results
    iterative_results = dict()
    iterative_results['general'] = pd.DataFrame()
    iterative_results['run_times'] = pd.DataFrame()
    iterative_results['performance'] = pd.DataFrame()
    iterative_results['model_size'] = pd.DataFrame()
    
    first_iter = True
    
    # For every iteration... (or once, if iterations are disabled)
    while Ab.setup.scenarios['samples'] < Ab.setup.scenarios['samples_max'] \
        or Ab.setup.main['iterative'] is False:
        
        if not first_iter:
            Ab.setup.scenarios['samples'] = \
                int(Ab.setup.scenarios['samples']*Ab.setup.scenarios['gamma'])
            case_id += 1
            
        # Shortcut to sample complexity
        N = Ab.setup.scenarios['samples']
        general_df = pd.DataFrame(data=N, index=[case_id], columns=['N'])        
            
        # Only perform code below is a new run is chosen
        if Ab.setup.main['newRun'] is True:
        
            print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            print('START ITERATION WITH SAMPLE COMPLEXITY N = '+str(N))
        
            # Set name for the seperate output folders of different instances
            Ab.setup.directories['outputFcase'] = \
                Ab.setup.directories['outputF'] + 'N='+str(N)+'/' 
            
            # Create folder to save results
            createDirectory( Ab.setup.directories['outputFcase'] )    
        
            # Compute factorial of the sample complexity upfront
            Ab.setup.scenarios['log_factorial_N'] = \
                math.log(math.factorial(Ab.setup.scenarios['samples']))
            
            # Calculate transition probabilities
            Ab.defTransitions()
            
            # Save case-specific data in Excel
            output_file = Ab.setup.directories['outputFcase'] + \
                Ab.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
            
            # Create a Pandas Excel writer using XlsxWriter as the engine
            writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
            
            # Build MDP
            model_size = Ab.buildMDP()
            
            # Write model size results to Excel
            model_size_df = pd.DataFrame(model_size, index=[case_id])
            model_size_df.to_excel(writer, sheet_name='Model size')
            
            # Solve the MDP
            Ab.solveMDP()
            
            # Write results to dataframes
            horizonLen = Ab.mdp.horizonLen
            
            # Load data into dataframes
            policy_df   = pd.DataFrame( Ab.results['policy']['action'][1], 
             columns=range(len(Ab.abstr['P'])), index=range(horizonLen)).T
            delta_df    = pd.DataFrame( Ab.results['policy']['delta'][1], 
             columns=range(len(Ab.abstr['P'])), index=range(horizonLen)).T
            reward_df   = pd.DataFrame( Ab.results['reward'].T, 
             columns=range(len(Ab.abstr['P'])), index=[0]).T
            
            # Write dataframes to a different worksheet
            policy_df.to_excel(writer, sheet_name='Optimal policy')
            delta_df.to_excel(writer, sheet_name='Optimal delta')
            reward_df.to_excel(writer, sheet_name='Optimal reward')
            
        # If no new run was chosen, load the results from existing data files
        else:
            
            # Load results from existing PRISM results files
            output_folder, policy_file, vector_file = load_PRISM_result_file(
                Ab.setup.main['mode_prefix'], Ab.setup.directories['output'], 
                Ab.system.name, N)
            
            # Retreive output folder
            Ab.setup.directories['outputFcase'] = output_folder
            
            print(' -- Load policy file:',policy_file)
            print(' -- Load vector file:',vector_file)
        
            # Save case-specific data in Excel
            output_file = Ab.setup.directories['outputFcase'] + \
                Ab.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
            
            # Create a Pandas Excel writer using XlsxWriter as the engine
            writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
            # Load results
            Ab.loadPRISMresults(policy_file, vector_file)
        
        # Initialize plotting
        Ab.preparePlots()
        
        if Ab.setup.montecarlo['enabled']:
            # Perform monte carlo simulation for validation purposes
            
            # setup.setOptions(category='montecarlo', init_states=[7])
            Ab.monteCarlo()
            
            # Store Monte Carlo results as dataframe
            cols = Ab.setup.montecarlo['init_timesteps']
            MCsims_df = pd.DataFrame( 
                Ab.mc['results']['reachability_probability'], \
                columns=cols, index=Ab.abstr['P'].keys())
                
            # Write Monte Carlo results to Excel
            MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
        
        # Plot results
        Ab.generatePlots( delta_value = Ab.setup.deltas[0], 
                           max_delta = max(Ab.setup.deltas),
                           case_id = case_id,
                           writer = writer)
        
        # %%
        
        # Store run times of current iterations        
        time_df = pd.DataFrame( data=Ab.time, index=[case_id] )
        time_df.to_excel(writer, sheet_name='Run time')
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        
        plt.close('all')
        
        # If iterative approach is enabled...
        if Ab.setup.main['iterative'] is True:
            
            # Add current results to general dictionary
            iterative_results['general'] = pd.concat(
                [iterative_results['general'], general_df], axis=0)
            iterative_results['run_times'] = pd.concat(
                [iterative_results['run_times'], time_df], axis=0)
            iterative_results['model_size'] = pd.concat(
                [iterative_results['model_size'], model_size_df], axis=0)
            
            first_iter = False
            
        else:
            print('\nITERATIVE SCHEME DISABLED, SO TERMINATE LOOP')
            break
    
    if Ab.setup.main['iterative'] and Ab.setup.main['newRun']:
        # Save overall data in Excel (for all iterations combined)
        output_file = Ab.setup.directories['outputF'] + \
            Ab.setup.time['datetime'] + '_iterative_results.xlsx'
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        for key,df in iterative_results.items():
            df.to_excel(writer, sheet_name=str(key))
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        
    return Ab