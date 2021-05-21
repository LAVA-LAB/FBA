#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:
    "Filter-based abstractions with correctness guarantees for planning under
    uncertainty"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>
______________________________________________________________________________
"""

import matplotlib.pyplot as plt # Import to generate plos using Pyplot
from datetime import datetime   # Import Datetime to retreive current date/time
import pandas as pd             # Import Pandas to store data in frames
import numpy as np
import os

latex_path = '/usr/local/texlive/2017/bin/x86_64-darwin'

if latex_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2017/bin/x86_64-darwin'

# Main classes and methods used
from core.scenarioBasedAbstraction import scenarioBasedAbstraction
from core.preprocessing.user_interface import user_choice
from core.commons import printWarning, createDirectory

from inspect import getmembers, isclass
from core import modelDefinitions

from core.masterClasses import settings

# Retreive a list of all available models
modelClasses = np.array(getmembers(modelDefinitions, isclass))
application, application_id  = user_choice('application',list(modelClasses[:,0]))

#np.random.seed(33)

#-----------------------------------------------------------------------------
# Create model class
#-----------------------------------------------------------------------------

# Create model object
model = modelClasses[application_id, 1]()

# Manual changes in model settings
model.setOptions(endTime=64)

#-----------------------------------------------------------------------------
# Create settings object + change manual settings
#-----------------------------------------------------------------------------

# Create settings object
setup = settings(mode='scenario', application=model.name)

# Manual changes in general settings
setup.setOptions(category='plotting', exportFormats=['pdf'], partitionPlot=True)
setup.setOptions(category='mdp', 
                 prism_java_memory=7,
                 prism_model_writer='explicit', # Is either default or explicit
                 prism_folder="/Users/thom/Documents/PRISM/prism-imc-v4/prism/") # Folder where PRISM is located
# setup.setOptions(category='montecarlo', init_states=[7])
    
setup.setOptions(category='main', iterative=False)
setup.setOptions(category='scenarios', samples=1600)

setup.setOptions(category='mdp', mode='interval')

# %%

#-----------------------------------------------------------------------------
# General settings (independent of application)
#-----------------------------------------------------------------------------

# If TRUE monte carlo simulations are performed
setup.montecarlo['enabled'], _ = user_choice( \
                                'Monte Carlo simulations', [True, False])
if setup.montecarlo['enabled']:
    setup.montecarlo['iterations'], _ = user_choice( \
                                'Monte Carlo iterations', 'integer')
else:
    setup.montecarlo['iterations'] = 0

print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('PROGRAM STARTED AT \n'+setup.time['datetime'])
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

#-----------------------------------------------------------------------------
# Initialize the main abstraction meethod
#-----------------------------------------------------------------------------

# Dictionary for all instances
ScAb = dict()

if setup.mdp['solver'] == 'Python' and setup.mdp['horizon'] == 'infinite':
    printWarning("Cannot solve infinite horizon MDP internally; switch to PRISM only")
    setup.mdp['solver'] = 'PRISM'

# Set case-specific parameters
setup.deltas = [2]

# Set LTI model in main object
model.setModel(observer=False)

# Create the main object for the current instance
ScAb = scenarioBasedAbstraction(setup=setup, basemodel=model)

# Remove initial variable dictionaries (reducing data usage)
del setup
del model

#-----------------------------------------------------------------------------
# Define actions (only required once outside iterative scheme)
#-----------------------------------------------------------------------------

# Create actions and determine which ones are enabled
ScAb.defActions()

import math

#-----------------------------------------------------------------------------
# Code below is repeated every iteration of the iterative scheme
#-----------------------------------------------------------------------------
case_id = 0

while ScAb.setup.scenarios['samples'] <= ScAb.setup.scenarios['samples_max'] or \
    ScAb.setup.main['iterative'] is False:

    # Shortcut to sample complexity
    N = ScAb.setup.scenarios['samples']        

    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('START ITERATION WITH SAMPLE COMPLEXITY N = '+str(N))

    # Set name for the seperate output folders of different instances
    ScAb.setup.directories['outputFcase'] = \
        ScAb.setup.directories['outputF'] + 'N='+str(N)+'/' 
    
    # Create folder to save results
    createDirectory( ScAb.setup.directories['outputFcase'] )    

    # Compute factorial of the sample complexity upfront
    ScAb.setup.scenarios['log_factorial_N'] = \
        math.log(math.factorial(ScAb.setup.scenarios['samples']))
    
    # Calculate transition probabilities
    ScAb.defTransitions()
    
    # Build MDP
    ScAb.buildMDP()
    
    # Save case-specific data in Excel
    output_file = ScAb.setup.directories['outputFcase'] + \
                        'N='+str(N)+'_data_export.xlsx'
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    time_df = pd.DataFrame( data=ScAb.time, index=[case_id] )
    
    time_df.to_excel(writer, sheet_name='Run time')
    
    # Solve the MDP
    ScAb.solveMDP()
    
    if ScAb.setup.montecarlo['enabled']:
        # Perform monte carlo simulation for validation purposes
        
        # setup.setOptions(category='montecarlo', init_states=[7])
        ScAb.monteCarlo()
        
        # Store Monte Carlo results as dataframe
        cols = ScAb.setup.montecarlo['init_timesteps']
        MCsims_df = pd.DataFrame( ScAb.mc['results']['reachability_probability'], \
                                 columns=cols, index=range(ScAb.abstr['nr_regions']) )
            
        # Write Monte Carlo results to Excel
        MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
    
    #%%
    
    # Plot results
    ScAb.generatePlots( delta_value = ScAb.setup.deltas[0], max_delta = max(ScAb.setup.deltas) )        
    
    # Write results to dataframes
    horizon_len = int(ScAb.N/min(ScAb.setup.deltas))
    
    # Load data into dataframes
    policy_df   = pd.DataFrame( ScAb.results['optimal_policy'], 
       columns=range(ScAb.abstr['nr_regions']), index=range(horizon_len) ).T
    delta_df    = pd.DataFrame( ScAb.results['optimal_delta'], 
       columns=range(ScAb.abstr['nr_regions']), index=range(horizon_len) ).T
    reward_df   = pd.DataFrame( ScAb.results['optimal_reward'], 
       columns=range(ScAb.abstr['nr_regions']), index=range(horizon_len) ).T
    
    # Write dataframes to a different worksheet
    policy_df.to_excel(writer, sheet_name='Optimal policy')
    delta_df.to_excel(writer, sheet_name='Optimal delta')
    reward_df.to_excel(writer, sheet_name='Optimal reward')
    
    # %%
    
    if ScAb.basemodel.name == 'UAV' or ScAb.basemodel.name == 'UAV_v2':
    
        from core.postprocessing.createPlots import UAVplot2D, UAVplot3D
        from core.mainFunctions import computeRegionCenters
        from core.commons import setStateBlock
        
        # Determine desired state IDs
        if ScAb.basemodel.name == 'UAV':
            if ScAb.basemodel.modelDim == 2:
                x_init = setStateBlock(ScAb.basemodel.setup['partition'], a=[-8], b=[0], c=[-8], d=[0])
                cut_value = [0,0]
                
            elif ScAb.basemodel.modelDim == 3:
                x_init = setStateBlock(ScAb.basemodel.setup['partition'], a=[-3], b=[0], c=[-3], d=[0], e=[-3], f=[0])
                cut_value = [0,0,0]
        
        elif ScAb.basemodel.name == 'UAV_v2':
            if ScAb.basemodel.modelDim == 2:
                x_init = setStateBlock(ScAb.basemodel.setup['partition'], a=[-6], b=[0], c=[-6], d=[0])
                cut_value = [1,1]
                
                
        # Compute all centers of regions associated with points
        x_init_centers = computeRegionCenters(np.array(x_init), ScAb.basemodel.setup['partition'])
        
        # Filter to only keep unique centers
        x_init_unique = np.unique(x_init_centers, axis=0)
        
        state_idxs = [ScAb.abstr['allCenters'][tuple(c)] for c in x_init_unique 
                                       if tuple(c) in ScAb.abstr['allCenters']]
        
        print(' -- Perform simulations for initial states:',state_idxs)
        
        ScAb.setup.montecarlo['init_states'] = state_idxs
        ScAb.setup.montecarlo['iterations'] = 100
        ScAb.monteCarlo()
        
        PRISM_reach = ScAb.results['optimal_reward'][0,state_idxs]
        empirical_reach = ScAb.mc['results']['reachability_probability'][state_idxs]
        
        print('Probabilistic reachability (PRISM): ',PRISM_reach)
        print('Empirical reachability (Monte Carlo):',empirical_reach)
        
        performance_df = pd.DataFrame( {'PRISM reachability': PRISM_reach.flatten(),
                                        'Empirical reachability': empirical_reach.flatten() } )
        performance_df.to_excel(writer, sheet_name='Performance')
        
        itersToShow = 10
        
        traces = []
        for i in state_idxs:
            for j in range(itersToShow):
                traces += [ScAb.mc['traces'][0][i][j]]
        
        min_delta = int(min(ScAb.setup.deltas))
        
        if ScAb.basemodel.modelDim == 2:
            UAVplot2D( ScAb.setup, ScAb.model[min_delta], ScAb.abstr, traces, cut_value )
        
        elif ScAb.basemodel.modelDim == 3:
            UAVplot3D( ScAb.setup, ScAb.model[min_delta], ScAb.abstr, traces, cut_value )
        
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
        
    # %%
    
    if ScAb.basemodel.name == 'building_2room' or (ScAb.basemodel.name == 'UAV' and ScAb.basemodel.modelDim == 2):
        
        import seaborn as sns
        from core.mainFunctions import definePartitions
        
        if ScAb.basemodel.name == 'building_2room':
        
            x_nr = ScAb.basemodel.setup['partition']['nrPerDim'][0]
            y_nr = ScAb.basemodel.setup['partition']['nrPerDim'][1]
            
            cut_centers = definePartitions(ScAb.basemodel.n, [x_nr, y_nr, 1, 1], 
                   ScAb.basemodel.setup['partition']['width'], 
                   ScAb.basemodel.setup['partition']['origin'], onlyCenter=True)
            
        elif ScAb.basemodel.name == 'UAV':
            
            x_nr = ScAb.basemodel.setup['partition']['nrPerDim'][0]
            y_nr = ScAb.basemodel.setup['partition']['nrPerDim'][2]
            
            cut_centers = definePartitions(ScAb.basemodel.n, [x_nr, 1, y_nr, 1], 
                   ScAb.basemodel.setup['partition']['width'], 
                   ScAb.basemodel.setup['partition']['origin'], onlyCenter=True)
                              
        cut_values = np.zeros((x_nr, y_nr))
        cut_coords = np.zeros((x_nr, y_nr, ScAb.basemodel.n))
        
        cut_idxs = [ScAb.abstr['allCenters'][tuple(c)] for c in cut_centers 
                                       if tuple(c) in ScAb.abstr['allCenters']]              
        
        for i,(idx,center) in enumerate(zip(cut_idxs, cut_centers)):
            
            j = i % y_nr
            k = i // y_nr
            
            cut_values[j,k] = ScAb.results['optimal_reward'][0,idx]
            cut_coords[j,k,:] = center
        
        cut_df = pd.DataFrame( cut_values, index=cut_coords[0,:,0], columns=cut_coords[:,0,1] )
        
        # %%
        
        fig = plt.figure()
        ax = sns.heatmap(cut_df, cmap="YlGnBu")
        ax.invert_yaxis()
        plt.show()
        
    # %%
    
    if ScAb.setup.main['iterative'] is True:
        ScAb.setup.scenarios['samples'] = int(ScAb.setup.scenarios['samples']*ScAb.setup.scenarios['gamma'])
        case_id += 1
        
    else:
        print('\nITERATIVE SCHEME DISABLED, SO TERMINATE LOOP')
        break
        
datestring_end = datetime.now().strftime("%m-%d-%Y %H-%M-%S")             
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datestring_end)