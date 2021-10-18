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

# Retreive a list of all available models
modelClasses = np.array(getmembers(modelDefinitions, isclass))
application, application_id  = user_choice('application',
                                           list(modelClasses[:,0]))

np.random.seed(10)

#-----------------------------------------------------------------------------
# Create model object
#-----------------------------------------------------------------------------

# Create model object
system = modelClasses[application_id, 1]()

#-----------------------------------------------------------------------------
# Create settings object + change manual settings
#-----------------------------------------------------------------------------

# Create settings object
setup = settings(application=system.name)

setup.setOptions(category='main',
        skewed=False,
        mode='Filter')

if setup.main['mode'] == 'Filter':
    # Filter-based abstraction
    setup.main['mode_prefix'] = 'FiAb'
    system.setModel(observer=True)
else:
    # Scenario-based abstraction
    setup.main['mode_prefix'] = 'ScAb'
    system.setModel(observer=False)
    
#-----------------------------------------------------------------------------
# Create new vs. load existing abstraction
#-----------------------------------------------------------------------------

# If TRUE create a new abstraction; load existing abstraction otherwise
_, choice = user_choice( \
    'Start a new abstraction or load existing PRISM results?', 
    ['New abstraction', 'Load existing results'])
setup.main['newRun'] = not choice

if setup.main['newRun']:
        
    # Let the user determine if 2-phase time horizon should be enabled
    twophase, _ = user_choice( 'Enable the 2-phase time horizon?', [True, False])    
    
    if twophase:    
        setup.mdp['k_steady_state'], _ = user_choice( \
                                        'value of k at which the steady state phase starts', 'integer')
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
setup.montecarlo['enabled'], _ = user_choice( \
                                'Monte Carlo simulations', [True, False])
if setup.montecarlo['enabled']:
    setup.montecarlo['iterations'], _ = user_choice( \
                                'Monte Carlo iterations', 'integer')
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
    
setup.directories['outputF']  = setup.directories['output']+setup.main['mode_prefix'] + \
    '_'+application+'_' + 'ksteadystate=' + str(setup.mdp['k_steady_state']) + \
    '_' + setup.time['datetime']+'/'
    
setup.floating_point_precision = 5
setup.MDP_prob_decimals = 5 # Number of decimals to round off on

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