#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ____________________________________
|                                    |
|  FILTER-BASED ABSTRACTION PROGRAM  |
|____________________________________|

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

# Default pyplot style (font size, template, etc.)
plt.close('all')
plt.ion()
plt.style.use('seaborn-deep')
plt.rcParams.update({'font.size': 7, 
                     'pgf.texsystem' : "xelatex"})

from inspect import getmembers, isclass
from core import modelDefinitions

from core.masterClasses import settings

# Retreive a list of all available models
modelClasses = np.array(getmembers(modelDefinitions, isclass))
application, application_id  = user_choice('application',list(modelClasses[:,0]))

#------------------------------------------------------------------------------
# PLOT FONT SIZES
#------------------------------------------------------------------------------

# Plot font family and size
plt.rc('font', family='serif')
SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 9

# Make sure the matplotlib generates editable text (e.g. for Illustrator)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set font sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#-----------------------------------------------------------------------------
# Main program setup
#-----------------------------------------------------------------------------

model = modelClasses[application_id, 1]()
setup = settings(mode='scenario', application=model.name)

# Manual changes in general settings
setup.setOptions(category='plotting', exportFormats=['pdf'], partitionPlot=False)
# setup.setOptions(category='montecarlo', init_states=[7])

# Manual changes in model settings
model.setOptions(endTime=64)
    
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
# Execute the main function
#-----------------------------------------------------------------------------

# Dictionary for all instances
ScAb = dict()
case_id = 0

# Retreive datetime string
ScAb['start_time'] = datetime.now().strftime("%H-%M-%S")

if 'Python' in setup.mdp['solver'] and setup.mdp['horizon'] == 'infinite':
    printWarning("Cannot solve infinite horizon MDP internally; switch to PRISM only")
    setup.mdp['solver'].remove('Python')

# Close all plots upon start of new case
plt.close('all')

# Set case-specific parameters
setup.deltas = [2]

# Set name for the seperate output folders of different instances
setup.directories['outputFcase'] = \
    setup.directories['outputF'] + 'case='+str(case_id)+'/' 

# Create folder to save results
createDirectory( setup.directories['outputFcase'] )

# Set LTI model in main object
model.setModel(tau=0.8, observer=False)

# Create the main object for the current instance
ScAb = scenarioBasedAbstraction(setup=setup, basemodel=model)

# Create actions and determine which ones are enabled
ScAb.defActions()

# Calculate transition probabilities
ScAb.defTransitions()

# Build MDP
ScAb.buildMDP()

# Save case-specific data in Excel
output_file = ScAb.setup.directories['outputFcase'] + \
                    'case='+str(case_id)+'_data_export.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

time_df = pd.DataFrame( data=ScAb.time, index=[case_id] )

time_df.to_excel(writer, sheet_name='Run time')

# %%

# Solve the MDP
ScAb.solveMDP()

# Shortcut to number of regions
S = ScAb.abstr['nr_regions']

if setup.montecarlo['enabled']:
    # Perform monte carlo simulation for validation purposes
    
    # setup.setOptions(category='montecarlo', init_states=[7])
    ScAb.monteCarlo()
    
    # Store Monte Carlo results as dataframe
    cols = ScAb.setup.montecarlo['init_timesteps']
    MCsims_df = pd.DataFrame( ScAb.mc['results']['reachability_probability'], \
                             columns=cols, index=range(S) )
        
    # Write Monte Carlo results to Excel
    MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
# Plot results
#%%

ScAb.generatePlots( delta_value = setup.deltas[0], max_delta = max(setup.deltas) )        

# Write results to dataframes
horizon_len = int(ScAb.N/min(ScAb.setup.deltas))
policy_df   = pd.DataFrame( ScAb.results['optimal_policy'], columns=range(S), index=range(horizon_len) )
delta_df    = pd.DataFrame( ScAb.results['optimal_delta'], columns=range(S), index=range(horizon_len) )
reward_df   = pd.DataFrame( ScAb.results['optimal_reward'], columns=range(S), index=range(horizon_len) )

# Write dataframes to a different worksheet
policy_df.to_excel(writer, sheet_name='Optimal policy')
delta_df.to_excel(writer, sheet_name='Optimal delta')
reward_df.to_excel(writer, sheet_name='Optimal reward')
    
# Close the Pandas Excel writer and output the Excel file.
writer.save()

# %%

datestring_end = datetime.now().strftime("%m-%d-%Y %H-%M-%S")             
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datestring_end)

# %%

if model.name == 'drone' and model.modelDim == 2:

    from core.postprocessing.createPlots import dronePositionPlot
    from core.mainFunctions import computeRegionCenters
    
    # Determine desired state IDs
    point_list = [[a,b,c,d] 
                for a in [-8,-6] 
                for b in [0]
                for c in [-8,-6] 
                for d in [0]]
    
    # Compute all centers of regions associated with points
    centers = computeRegionCenters(np.array(point_list), ScAb.basemodel.setup['partition'])
    
    # Filter to only keep unique centers
    centers_unique = np.unique(centers, axis=0)
    
    state_idxs = [ScAb.abstr['allCenters'][tuple(c)] for c in centers_unique 
                                   if tuple(c) in ScAb.abstr['allCenters']]
    
    print(' -- Perform simulations for initial states:',state_idxs)
    
    ScAb.setup.montecarlo['init_states'] = state_idxs
    ScAb.setup.montecarlo['iterations'] = 3
    ScAb.monteCarlo()
    
    traces = []
    for i in state_idxs:
        traces += [ScAb.mc['traces'][0][i][0]]
    
    min_delta = int(min(ScAb.setup.deltas))
    dronePositionPlot( ScAb.setup, ScAb.model[min_delta], ScAb.abstr, traces )