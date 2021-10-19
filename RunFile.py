#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|   FILTER-BASED ABSTRACTION PROGRAM   |
|______________________________________|

Implementation of the method proposed in the paper:
 "Filter-Based Abstractions for Safe Planning of Partially Observable 
  Autonomous Systems"

Originally coded by:        Thom S. Badings
Contact e-mail address:     thom.badings@ru.nl
______________________________________________________________________________
"""

# Load general packages
from datetime import datetime   # Import Datetime to get current date/time
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
from inspect import getmembers, isclass # To get list of all available models
import pickle

# Load main classes and methods
from core.scenarioBasedAbstraction import scenarioBasedAbstraction
from core.filterBasedAbstraction import filterBasedAbstraction
from core.preprocessing.user_interface import user_choice, \
    load_PRISM_result_file
from core.commons import printWarning, createDirectory
from core import modelDefinitions
from core.masterClasses import settings, loadOptions

for wFactor in [1]:
  for vFactor in [0.1, 0.5, 1, 2]:

    #-----------------------------------------------------------------------------
    # Set override settings (preset)
    #-----------------------------------------------------------------------------
    
    preset = {
        'application_id': 1,
        'scenario': 3,          # Planning scenario for 3D UAV case
        'wFactor': wFactor,           # Process noise strength for UAV cases
        'vFactor': vFactor,           # Measurement noise strength for UAV cases
        'new': True,
        '2phase': True,
        'k_steady_state': 3,
        'montecarlo': False,
        'mc_iterations': 0,
        'partition_size_id': 0, # For double integrator and 1-room building
        'adaptive_rate': [2]    # Adaptive rates for double integrator case
        }
    
    #-----------------------------------------------------------------------------
    # Load model classes and set random seed
    #-----------------------------------------------------------------------------
    
    # Retreive a list of all available models
    modelClasses = np.array(getmembers(modelDefinitions, isclass))
    
    if 'application_id' not in preset:
        
        _, preset['application_id']  = user_choice('application',
                                                   list(modelClasses[:,0]))
    
    np.random.seed(10)
    
    #-----------------------------------------------------------------------------
    # Create model object
    #-----------------------------------------------------------------------------
    
    # Create model object
    system = modelClasses[preset['application_id'], 1](preset)
    
    #-----------------------------------------------------------------------------
    # Create settings object + change manual settings
    #-----------------------------------------------------------------------------
    
    # Create settings object
    setup = settings()
    
    setup.setOptions(category='main',
            skewed=False,
            mode='Filter')
    
    if setup.main['mode'] == 'Filter':
        # Filter-based abstraction
        setup.main['mode_prefix'] = 'FiAb'
        system.setModel(observer=True, preset=preset)
    else:
        # Scenario-based abstraction
        setup.main['mode_prefix'] = 'ScAb'
        system.setModel(observer=False, preset=preset)
        
    #-----------------------------------------------------------------------------
    # Create new vs. load existing abstraction
    #-----------------------------------------------------------------------------
    
    # If TRUE create a new abstraction; load existing abstraction otherwise
    if 'new' not in preset:
        _, choice = user_choice( \
            'Start a new abstraction or load existing PRISM results?', 
            ['New abstraction', 'Load existing results'])
        setup.main['newRun'] = not choice
    else:
        setup.main['newRun'] = preset['new']
    
    #-----------------------------------------------------------------------------
    # Other settings
    #-----------------------------------------------------------------------------
    
    if setup.main['newRun']:
            
        # Let the user determine if 2-phase time horizon should be enabled
        if '2phase' not in preset:
            preset['2phase'], _ = user_choice( 'Enable the 2-phase time horizon?', [True, False])    
        
        if preset['2phase']:   
            if 'k_steady_state' not in preset:
                setup.mdp['k_steady_state'], _ = user_choice( \
                                                'value of k at which the steady state phase starts', 'integer')
            else:
                setup.mdp['k_steady_state'] = preset['k_steady_state']
        else:
            setup.mdp['k_steady_state'] = None
            
    else:
        
        # Load results from existing PRISM results files
        output_folder, _, _ = load_PRISM_result_file(
            setup.main['mode_prefix'], setup.directories['output'], 
            system.name)
        
        with open(output_folder+'setup.pickle', 'rb') as handle:
            setup = pickle.load(handle)
            
        # Overwrite setting that we're not creating a new abstraction
        setup.main['newRun'] = False
        
        setup.directories['outputFcase'] = output_folder
    
    # If TRUE monte carlo simulations are performed
    if 'montecarlo' not in preset:
        setup.montecarlo['enabled'], _ = user_choice( \
                                        'Monte Carlo simulations', [True, False])
    else:
        setup.montecarlo['enabled'] = preset['montecarlo']
        
    if setup.montecarlo['enabled']:
        if 'mc_iterations' not in preset:
            setup.montecarlo['iterations'], _ = user_choice( \
                                    'Monte Carlo iterations', 'integer')
        else:
            setup.montecarlo['iterations'] = preset['mc_iterations']
    else:
        setup.montecarlo['iterations'] = 0
    
    # %%
    
    loadOptions('options.txt', setup)
    
    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('PROGRAM STARTED AT \n'+setup.time['datetime'])
    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    
    #-----------------------------------------------------------------------------
    # Initialize the main abstraction method
    #-----------------------------------------------------------------------------
    
    if setup.main['iterative'] is False:
        setup.scenarios['samples_max'] = setup.scenarios['samples']
    
    # Dictionary for all instances
    Ab = dict()
        
    model_prefix = ''
    
    if system.name in ['UAV_2D', 'UAV_3D']:
        model_prefix += '_wFact='+str(system.LTI['noise']['wFactor'])
        
        if setup.main['mode'] == 'Filter':
            model_prefix += '_vFact='+str(system.LTI['noise']['vFactor'])
            
    if system.name == 'UAV_3D':
        model_prefix += '_layout='+str(system.scenario)
                     

    setup.directories['outputF']  = setup.directories['output']+setup.main['mode_prefix'] + \
        '_'+system.name+'_' + 'ksteadystate=' + str(setup.mdp['k_steady_state']) + \
        model_prefix + '_' + setup.time['datetime']+'/'
        
    # Number of decimals to round off on
    setup.floating_point_precision = 5
    setup.MDP_prob_decimals = 5
    
    '''
    if setup.main['newRun'] is True:
        # Create noise samples
        if system.name in ['UAV'] and system.modelDim == 3:
            setup.setOptions(category='scenarios', gaussian=False)
            system.setTurbulenceNoise(setup.scenarios['samples_max'])
    '''
    
    if setup.main['iterative'] is True and setup.main['newRun'] is False:
        printWarning("Iterative scheme cannot be combined with loading existing "+
                     "PRISM results, so iterative scheme disabled")
        setup.main['iterative'] = False
        setup.scenarios['samples_max'] = setup.scenarios['samples']
    
    # Create the main object for the current instance
    if setup.main['mode'] == 'Filter':
        Ab = filterBasedAbstraction(setup=setup, system=system)
        
        # Set name for the seperate output folders of different instances
        if Ab.setup.main['newRun']:
            Ab.setup.directories['outputFcase'] = \
                Ab.setup.directories['outputF']    
        
    else:
        Ab = scenarioBasedAbstraction(setup=setup, system=system)
    
    # Remove initial variable dictionaries (reducing data usage)
    del setup
    del system
    
    #-----------------------------------------------------------------------------
    # Define actions (only required once outside iterative scheme)
    #-----------------------------------------------------------------------------
    
    if Ab.setup.main['newRun'] is True:        
    
        # Create directories
        createDirectory(Ab.setup.directories['outputF'])    
    
        with open(Ab.setup.directories['outputF']+'setup.pickle', 'wb') as handle:
            pickle.dump(Ab.setup, handle)
    
        # Create actions and determine which ones are enabled
        Ab.defineActions()
        
    else:
        # If no new run was chosen, load the results from existing data files
        
        print(' -- Initialize MDP object')
        
        from core.createMDP import mdp
        Ab.mdp = mdp(Ab.setup, Ab.N, Ab.abstr)
        
        print(' -- Load dataframe from .json file')
        
        Ab.mdp.MAIN_DF = pd.read_json(Ab.setup.directories['outputFcase']+'output_dataframe.json')
        
        # Save case-specific data in Excel
        output_file = Ab.setup.directories['outputFcase'] + \
            Ab.setup.time['datetime'] + '_data_export.xlsx'
    
    #-----------------------------------------------------------------------------
    
    if Ab.setup.main['mode'] == 'Filter':
        
        from FiAb import filterBasedScheme    
    
        Ab = filterBasedScheme(Ab, case_id=0)
    
    else:
        
        from ScAb import iterativeScheme
        
        Ab = iterativeScheme(Ab)
        
    datestring_end = datetime.now().strftime("%m-%d-%Y %H-%M-%S")             
    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('APPLICATION FINISHED AT', datestring_end)