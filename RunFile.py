#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:
 "Sampling-Based Robust Control of Autonomous Systems with Non-Gaussian Noise"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>
______________________________________________________________________________
"""

# Load general packages
from datetime import datetime   # Import Datetime to get current date/time
import pandas as pd             # Import Pandas to store data in frames
import numpy as np              # Import Numpy for computations
import math                     # Import Math for mathematical operations
import matplotlib.pyplot as plt # Import Pyplot to generate plots

from inspect import getmembers, isclass # To get list of all available models

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
setup.base_delta = system.base_delta

loadOptions('options.txt', setup)

# %%

# Manual changes in general settings
setup.setOptions(category='plotting', 
        exportFormats=['pdf'], 
        partitionPlot=False,
        partitionPlot_plotHull=True)

setup.setOptions(category='scenarios',
        samples=400)

setup.setOptions(category='main',
        skewed=False,
        mode='Filter')

if setup.main['mode'] == 'Filter':
    setup.main['mode_prefix'] = 'FiAb'
else:
    setup.main['mode_prefix'] = 'ScAb'

setup.directories['outputF']  = setup.directories['output']+setup.main['mode_prefix']+ \
    '_'+application+'_' + setup.time['datetime']+'/'

print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

#-----------------------------------------------------------------------------
# Settings related to Monte Carlo simulations
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
# Initialize the main abstraction method
#-----------------------------------------------------------------------------

if setup.main['iterative'] is False:
    setup.scenarios['samples_max'] = setup.scenarios['samples']

# Dictionary for all instances
Ab = dict()

# Set LTI model in main object
if setup.main['mode'] == 'Filter':
    system.setModel(observer=True)
else:
    system.setModel(observer=False)
    
setup.lic = {'jump_factors': [2,4]}
setup.mdp['k_steady_state'] = 4
setup.main['covarianceMode'] = ['SDP','iterative'][0]
setup.main['interval_margin'] = 0.001
setup.precision = 5

# If TRUE monte carlo simulations are performed
_, choice = user_choice( \
    'Start a new abstraction or load existing PRISM results?', 
    ['New abstraction', 'Load existing results'])
setup.main['newRun'] = not choice

if setup.main['newRun'] is True:
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

    # Create actions and determine which ones are enabled
    Ab.defineActions()

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