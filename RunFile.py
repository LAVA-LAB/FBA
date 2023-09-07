#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|   FILTER-BASED ABSTRACTION PROGRAM   |
|______________________________________|

Implementation of the method proposed in the paper:
 "Filter-Based Abstractions for Safe Planning of Partially Observable 
  Dynamical Systems"

Originally coded by:        Thom S. Badings
Contact e-mail address:     thom.badings@ru.nl
______________________________________________________________________________
"""

# Load general packages
from datetime import datetime   # Import Datetime to get current date/time
from tabulate import tabulate
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
from inspect import getmembers, isclass # To get list of all available models
import pickle

# Load main classes and methods
from core.scenarioBasedAbstraction import scenarioBasedAbstraction
from core.filterBasedAbstraction import filterBasedAbstraction
from core.preprocessing.user_interface import user_choice, \
    load_PRISM_result_file, parse_arguments
from core.commons import printWarning, createDirectory
from core import modelDefinitions
from core.masterClasses import settings, loadOptions

preset = parse_arguments(run_in_vscode = False)

# preset.application = "spacecraft"
# preset.two_phase_transient_length = 1
# preset.monte_carlo_iterations = -1
# preset.R_size = [11, 11, 5, 5]
# preset.R_width = [2, 2, 0.05, 0.05]
# preset.plot_heatmap = [0, 1]
# preset.plot_trajectory_2D = [0, 1]
# preset.noise_strength_w = [1, 1, 1, 1]
# preset.noise_strength_v = [1, 1]
# preset.validate_performance = 100
# preset.horizon = 32

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(preset).items(), headers=["Argument", "Value"]),'\n')

#-----------------------------------------------------------------------------
# Load model classes and set random seed
#-----------------------------------------------------------------------------

# Retreive a list of all available models
modelClasses = np.array(getmembers(modelDefinitions, isclass))

if preset.application == -1:
    preset.application, _  = user_choice('application',
                                               list(modelClasses[:,0]))

np.random.seed(10)

#-----------------------------------------------------------------------------
# Create model object
#-----------------------------------------------------------------------------

# Create model object
application_id = np.where(modelClasses[:,0] == preset.application)[0]
assert len(application_id) == 1
system = modelClasses[application_id[0], 1](preset)

#-----------------------------------------------------------------------------
# Create settings object + change manual settings
#-----------------------------------------------------------------------------

# Create settings object
setup = settings()
setup.preset = preset

setup.setOptions(category='main',
        skewed=False,
        mode='Filter',
        error_bound_beta = 0.01)

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
setup.main['newRun'] = not preset.load_results

#-----------------------------------------------------------------------------
# Other settings
#-----------------------------------------------------------------------------

if setup.main['newRun']:
        
    # Disable two-phase horizon
    setup.mdp['k_steady_state'] = -1 if preset.two_phase_transient_length == -1 else \
                                    preset.two_phase_transient_length
     
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
setup.montecarlo['enabled'] = False if preset.monte_carlo_iterations == -1 else True
setup.montecarlo['iterations'] = preset.monte_carlo_iterations

loadOptions('options.txt', setup)

if setup.mdp['mode'] == 'estimate' and len(system.adaptive['rates']) > 0:
    
    printWarning('Warning: "estimate" mode (MDP) is incompatible with adaptive measurement scheme, so switch to "interval" mode (iMDP)')

    setup.mdp['mode'] = 'interval'

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
    model_prefix += '_wFact='+str(system.LTI['noise']['noise_strength_w'])
    
    if setup.main['mode'] == 'Filter':
        model_prefix += '_vFact='+str(system.LTI['noise']['noise_strength_v'])
        
if system.name == 'UAV_3D':
    model_prefix += '_layout='+str(system.scenario)
                 

setup.directories['outputF']  = setup.directories['output']+setup.main['mode_prefix'] + \
    '_'+system.name+'_' + 'ksteadystate=' + str(setup.mdp['k_steady_state']) + \
    model_prefix + '_' + setup.time['datetime']+'/'
    
# Number of decimals to round off on
setup.floating_point_precision = 5
setup.MDP_prob_decimals = 5

if setup.main['mode'] != 'Filter' and setup.main['newRun'] is True:
    # Create noise samples
    if system.name in ['UAV'] and system.modelDim == 3:
        setup.setOptions(category='scenarios', gaussian=False)
        system.setTurbulenceNoise(setup.scenarios['samples_max'])

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
# del setup
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