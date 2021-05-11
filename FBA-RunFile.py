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
import os                       # Import OS to allow creationg of folders
import pandas as pd             # Import Pandas to store data in frames
import numpy as np
import math

# Main classes and methods used
from core.filterBasedAbstraction import filterBasedAbstraction
from core.preprocessing.user_interface import user_choice
from core.commons import printWarning

# Default pyplot style (font size, template, etc.)
plt.close('all')
plt.ion()
plt.style.use('seaborn-deep')
plt.rcParams.update({'font.size': 7})

applications = ['motion-planning', '2zone-thermal']
application  = user_choice('application',applications)

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

setup                       = dict()
setup['folder']             = dict()
experiment                  = dict()
                                
# PRISM settings
setup['folder']['PRISM_filename'] = 'FBA-MDP'
setup['folder']['PRISM_modes'] = ['default','interval']

# If TRUE, additional outputs are printed
setup['verbose']            = True

# If TRUE, scenario-based probability bounds are determined and used
# Nr of samples considered for scenario-based probabilities is also given
setup['sa']                 = dict()
setup['sa']['switch']       = False
setup['sa']['samples']      = 100 # N
setup['sa']['confidence']   = 1e-1 #beta value
setup['sa']['log_factorial_N'] = math.log(math.factorial(setup['sa']['samples']))
setup['sa']['usetable']     = 'input/probabilityTable_N='+ \
                        str(setup['sa']['samples'])+'_beta='+ \
                        str(setup['sa']['confidence'])+'_d=1.csv'
    
if application == 'motion-planning':
    
    # List of MDP solving methods
    setup['MDPsolver']      = ['Python', 'PRISM']
    
    # Define case numbers
    experiment['instances'] = [0,1,2]
    
    # End time for simulation (last action can be chosen at 'endTime - tau')
    setup['endTime']        = 32
    
    # Authority limit for the control u, both positive and negative
    setup['uMin']           = [-5]
    setup['uMax']           = [5]
    
    # ID of the model type that is called
    setup['modelType']      = '2d'
    
    experiment['deltas']    = [[2], [8], [2,4,6,8]]
    
    # If TRUE, a threshold on the covariance of the belief is maintained, using
    # a 'local information controller' (funnel on covariance at every node)z
    experiment['LIC']       = [False, False, True]
    
    setup['LICMaxA']        = 1.5
    setup['LICMinA']        = 1.5

    # Number of partitions in each dimension from the center state
    setup['PartitionsFromCenter']   = 10
    
    # Maximum value of the state-space domain (both positive and negative)
    setup['stateSpaceDomainMax']    = 21
    setup['stateSpaceOrigin']       = [0, 0]

    # Initial Kalman filter covariance
    setup['kf_cov0']        = 10
    
    # Covariance values of the process noise (w) and measurement noise (v)
    setup['sigma_w_value']  = 0.15
    setup['sigma_v_value']  = 0.15
    
    # Partition plot switch
    setup['partitionPlot']  = True
    
    # Set which Monte Carlo simulations to perform (False means inherent)
    setup['MC_initS']       = False
    setup['MC_kStart']      = False
    
    setup['critical_state_center'] = [20]

elif application == '2zone-thermal':
    
    # List of MDP solving methods
    setup['MDPsolver']      = ['Python', 'PRISM']
    
    # Define case numbers
    experiment['instances']      = [1] #[0,1,2,3,4,5]
    experiment['deltas']         = [[2]]*6
    experiment['partition_size'] = [1, 3, 5, 7, 9, 11]
    
    # If TRUE, a threshold on the covariance of the belief is maintained, using
    # a 'local information controller' (funnel on covariance at every node)z
    experiment['LIC']       = [False]*6
    
    # ID of the model type that is called
    setup['modelType']      = 'thermal2zone'
    
    # Authority limit for the control u, both positive and negative
    setup['uMin']           = [0,0]# Watt
    setup['uMax']           = [1500,1500] # Watt
    
    # End time for simulation (last action can be chosen at 'endTime - tau')
    setup['endTime']        = 20
    
    setup['LICMaxA']        = 0.3
    setup['LICMinA']        = 0.3

    # Initial Kalman filter covariance
    setup['kf_cov0']        = 1
    
    # Covariance values of the process noise (w) and measurement noise (v)
    setup['sigma_w_value']  = 0.05
    setup['sigma_v_value']  = 0.5
    
    # Partition plot switch
    setup['partitionPlot']  = False
    
#-----------------------------------------------------------------------------
# General settings (independent of application)
#-----------------------------------------------------------------------------

# If TRUE monte carlo simulations are performed
setup['MCsims']   = user_choice('Monte Carlo simulations', [True, False])
if setup['MCsims']:
    setup['MCiter'] = user_choice('Monte Carlo iterations', 'integer')
else:
    setup['MCiter'] = 0

# Sampling time
setup['samplingTime']       = 1

# The interval margin determines the width (from mean approximation) for the
# induces intervals on the transition probabilities (for the iMDP)
setup['interval_margin']    = 0.01

# If TRUE, the target point of each action is determined as the point in the
# hull nearest to the actual goal point (typically the center state)
setup['dynamicTargetPoint']      = False

# If TRUE, the distance between the center of regions is linked to the 
# probability integrals. Note, this requires symmetric regions, with the
# target point being the center.
setup['efficientProbIntegral']   = True

# TRUE/FALSE setup whether plots should be generated
setup['partitionPlot_plotHull']  = True
setup['probabilityPlots']        = True
setup['MCtrajectoryPlot']        = False

# Retreive datetime string
setup['main_date'] = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

# Retreive working folder
setup['folder']['base']     = os.getcwd()
setup['folder']['output']   = setup['folder']['base']+'/Output/'
setup['folder']['outputF']  = setup['folder']['output']+'FiAb '+application+' '+ \
                                setup['main_date']+'/'

print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('PROGRAM STARTED AT \n'+datetime.now().strftime("%m-%d-%Y %H-%M-%S"))
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

#-----------------------------------------------------------------------------
# Execute the main function
#-----------------------------------------------------------------------------

# Create empty dataframe for storing timer results
experiment['time'] = pd.DataFrame()
    
# Create folder to save results
if not os.path.exists(setup['folder']['outputF']):
    os.makedirs(setup['folder']['outputF'])

# Dictionary for all instances
M = dict()

# Retreive datetime string
M['start_time'] = datetime.now().strftime("%H-%M-%S")

# For all instances in experiment
for i in experiment['instances']:
    
    # Close all plots upon start of new case
    plt.close('all')
    
    # Set case-specific parameters
    setup['LIC']        = experiment['LIC'][i]
    setup['deltas']     = experiment['deltas'][i]
    
    if application == '2zone-thermal':
        # For the 2-zone thermal problem, we may want to vary the 
        # partitioning of the state space
        
        # Number of partitions in each dimension from the center state
        n = experiment['partition_size'][i]
        temp_origin = -5
        
        setup['PartitionsFromCenter']   = n
        
        # Maximum value of the state-space domain (both positive and negative)
        setup['stateSpaceDomainMax']    = 7
        setup['stateSpaceOrigin']       = [temp_origin,temp_origin,0]
        
        # Goal temperature (for zones)
        width = setup['stateSpaceDomainMax'] / (n+0.5)
        setup['goal_state_center'] = [temp_origin-(n-1)*width]*2
        
        # Set which Monte Carlo simulations to perform (False means inherent)
        setup['MC_initS']          = [temp_origin+(n-1)*width]*2
        setup['MC_kStart']         = [0]

    # Set name for the seperate output folders of different instances
    setup['folder']['outputFcase'] = setup['folder']['outputF'] + \
                                            'case='+str(i)+'/' 

    # Create folder to save results
    if not os.path.exists(setup['folder']['outputFcase']):
        os.makedirs(setup['folder']['outputFcase'])

    # Create the main object for the current instance
    M[i] = filterBasedAbstraction(setup)
    
    # Perform Kalman filtering
    M[i].KalmanPrediction()
    
    # Calculate transition probabilities
    M[i].calculateTransitions()
    
    # Build MDP
    M[i].buildMDP()
    
    # Save case-specific data in Excel
    output_file = M[i].setup['folder']['outputFcase'] + \
                        'case='+str(i)+'_data_export.xlsx'
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    time_df = pd.DataFrame( data=M[i].time, index=[i] )
    experiment['time'] = experiment['time'].append( time_df )
    
    time_df.to_excel(writer, sheet_name='Run time')
    
    # Only continue if Python is in the list of MDP solving methods
    # And only do this if scenario approach is disabled
    if 'Python' in setup['MDPsolver']:
    
        if setup['sa']['switch']:    
            # del M[i]
            
            printWarning('Scenario approach active; cannot solve iMDP internally')
            
        else:
    
            # Solve the MDP using the internal solver
            M[i].solveMDP()
            
            # Shortcut to number of regions
            S = M[i].abstr['nr_regions']
            
            if setup['MCsims']:
                # Perform monte carlo simulation for validation purposes
                M[i].monteCarlo()
                
                # Store Monte Carlo results as dataframe
                cols = M[i].mc['setup']['start_times']
                MCsims_df = pd.DataFrame( M[i].mc['results']['reachability_probability'], \
                                         columns=cols, index=range(S) )
                    
                # Write Monte Carlo results to Excel
                MCsims_df.to_excel(writer, sheet_name='Empirical reach.')
        
            # Plot results
            M[i].generatePlots( delta_value = setup['deltas'][0], max_delta = max(experiment['deltas'][2]) )        
            
            # Write results to dataframes
            policy_df   = pd.DataFrame( M[i].results['optimal_policy'], columns=range(S), index=range(M[i].N) )
            delta_df    = pd.DataFrame( M[i].results['optimal_delta'], columns=range(S), index=range(M[i].N) )
            reward_df   = pd.DataFrame( M[i].results['optimal_reward'], columns=range(S), index=range(M[i].N) )
            
            # Write dataframes to a different worksheet
            policy_df.to_excel(writer, sheet_name='Optimal policy')
            delta_df.to_excel(writer, sheet_name='Optimal delta')
            reward_df.to_excel(writer, sheet_name='Optimal reward')
        
    else:
        
        # Delete object of previous case to save memory
        del M[i]
        
        printWarning('Python not in the list of MDP solving methods, so terminate\n')
        
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
#-----------------------------------------------------------------------------
# %% Analyzing the case results
#-----------------------------------------------------------------------------

from core.postprocessing.analyzeExperiment import analyzeExperiment

# Analyze experiment if Python solver is enabled, and nr instances is > 1
if len(experiment['instances']) > 1:
    experiment = analyzeExperiment(M, experiment, setup)

datestring_end = datetime.now().strftime("%m-%d-%Y %H-%M-%S")             
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print('APPLICATION FINISHED AT', datestring_end)

# %%

from core.postprocessing.plotTrajectory import confidence_ellipse 