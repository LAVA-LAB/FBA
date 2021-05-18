# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:03:21 2021

@author: Thom Badings
"""

import os                       # Import OS to allow creationg of folders

from datetime import datetime   # Import Datetime to retreive current date/time
from core.commons import createDirectory

class settings(object):
    
    def setOptions(self, category=None, **kwargs):
        # Function to set values in the setup dictionary

        # If category is none, settings are set in main dictionary
        if category==None:
            for key, value in kwargs.items():
                self.key = value
                
                print(' >> Changed',key,'to',value)
                
        # Otherwise, settings are set within sub-dictionary 'category'
        else:
            
            category_upd = getattr(self, category)
            
            for key, value in kwargs.items():
                
                category_upd[str(key)] = value
                
                print(' >> Changed',key,'in',category,'to',value)
                
            setattr(self, category, category_upd)
    
    def __init__(self, mode, application):
        
        # Default scenario approach settings
        sa = dict()
        gaussian = dict()
        
        # Mode is either 'scenario' or 'gaussian'
        if mode == 'scenario':
            sa['switch'] = True        
            sa['samples'] = 25 # Sample complexity used in scenario approach
            sa['gamma'] = 2 # Factor by which N is multiplied in every iteration
            sa['samples_max'] = 800 # Maximum number of samples in iterative scheme
            sa['confidence']   = 1e-1 # Confidence level (beta)
            
        else:
            sa['switch'] = False
            
            # The interval margin determines the width (from mean approximation) for the
            # induces intervals on the transition probabilities (for the iMDP)
            gaussian['margin'] = 0.01
            
            # If TRUE, the distance between the center of regions is linked to the 
            # probability integrals. Note, this requires symmetric regions, with the
            # target point being the center.
            gaussian['efficientIntegral'] = True

        
        # Default MDP and prism settings
        mdp = dict()
        mdp['filename'] = 'Abstraction'
        mdp['mode'] = ['estimate','interval'][1]
        mdp['horizon'] = ['infinite','finite'][0]
        mdp['solver'] = ['PRISM', 'Python'][0]
        mdp['prism_java_memory'] = 1 # PRISM java memory allocation in GB
        mdp['prism_model_writer'] = ['default','explicit'][1]
        mdp['prism_folder'] = "/Users/..."
        
        # Default time/date settings
        timing = dict()
        # Retreive datetime string
        timing['datetime'] = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        
        # Default folder/directory settings
        directories = dict()
        # Retreive working folder
        directories['base']     = os.getcwd()
        directories['output']   = directories['base']+'/output/'
        directories['outputF']  = directories['output']+'ScAb_'+application+'_'+ \
                                        timing['datetime']+'/'
        
        # Default plotting settings
        plot = dict()
        # TRUE/FALSE setup whether plots should be generated
        plot['partitionPlot']           = False
        plot['partitionPlot_plotHull']  = True
        plot['probabilityPlots']        = True
        plot['MCtrajectoryPlot']        = False
        plot['exportFormats']           = ['pdf','png']
        
        # Default Monte Carelo settings
        # Set which Monte Carlo simulations to perform (False means inherent)
        mc = dict()
        mc['init_states']           = False
        mc['init_timesteps']        = False
        
        self.mdp = mdp
        self.plotting = plot
        self.montecarlo = mc
        self.gaussian = gaussian
        self.scenarios = sa
        self.time = timing
        self.directories = directories
        
        self.verbose = True
        
        # Create directories
        createDirectory(self.directories['outputF'])

class LTI_master(object):
    
    def setOptions(self, category=None, **kwargs):
        # Function to set values in the setup dictionary

        # If category is none, settings are set in main dictionary
        if category==None:
            for key, value in kwargs.items():
                self.setup[str(key)] = value
                
                print(' >> Changed',key,'to',value)
                
        # Otherwise, settings are set within sub-dictionary 'category'
        else:
            for key, value in kwargs.items():
                self.setup[str(category)][str(key)] = value
    
                print(' >> Changed',key,'in',category,'to',value)
    
    def __init__(self):
        
        partition = dict()
        specification = dict()
        
        targets = dict()
       
        # If TRUE, the target point of each action is determined as the point in the
        # hull nearest to the actual goal point (typically the center state)
        targets['dynamic'] = False
        
        noise = dict()
        control = dict()
        control['limits'] = dict()
        
        self.setup = {
                'partition' :       partition,
                'specification' :   specification,
                'targets' :         targets,
                'noise' :           noise,
                'control' :         control,
                #
                'endTime' :         32
            }
        
        self.name = type(self).__name__