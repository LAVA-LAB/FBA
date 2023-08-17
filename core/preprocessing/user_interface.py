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

import glob
import os
import sys

import argparse
from ast import literal_eval

def parse_arguments(run_in_vscode):
    """
    Function to parse arguments provided

    Parameters
    ----------
    :manualModel: Override model as provided as argument in the command
    :nobisim: Override bisimulatoin option as provided as argument in the command

    Returns
    -------
    :args: Dictionary with all arguments

    """
    
    if run_in_vscode:
        sys.argv = [""]

    parser = argparse.ArgumentParser(description="Filter-based abstractions programme")
    
    parser.add_argument('--application', type=str, action="store", dest='application',
                        default=-1, help="Application/class name")
    
    parser.add_argument('--load_results', dest='load_results', action='store_true',
                        help="If true, results from a previous run are loaded")
    parser.set_defaults(load_results=False)

    parser.add_argument('--horizon', type=int, action="store", dest='horizon', 
                        default=10, help="Time horizon (nr discrete steps) for the reach-avoid problem")

    parser.add_argument('--two_phase_transient_length', type=int, action="store", dest='two_phase_transient_length', 
                        default=-1, help="Length of the transient phase of the 2-phase horizon (if -1, 2-phase horizon is not enabled)")
    
    parser.add_argument('--monte_carlo_iterations', type=int, action="store", dest='monte_carlo_iterations', 
                        default=-1, help="Number of Monte Carlo iterations (per initial conditions; if -1 is given, no MC is performed)")

    parser.add_argument('--adaptive_rate', dest='adaptive_rate', nargs='+', 
                        help='List of adaptive rates to use', default=[])
    
    parser.add_argument('--R_size', dest='R_size', nargs='+', 
                        help='Partition: number of regions per dimension', default=[])
    
    parser.add_argument('--R_width', dest='R_width', nargs='+', 
                        help='Partition: width of each region per dimension', default=[])

    parser.add_argument('--noise_strength_w', dest='noise_strength_w', nargs='+',
                        help='Multiplier for the process noise', default=[1])

    parser.add_argument('--noise_strength_v', dest='noise_strength_v', nargs='+',
                        help='Multiplier for the measurement noise', default=[1])

    # 3D UAV settings
    parser.add_argument('--scenario', type=int, action="store", dest='scenario',
                        default=3, help="Planning scenario")

    # Plot functions
    parser.add_argument('--plot_heatmap', dest='plot_heatmap', nargs='+', 
                        help='Plot heatmap for the two provided state variables', default=False)
    
    parser.add_argument('--plot_trajectory_2D', dest='plot_trajectory_2D', nargs='+', 
                        help='Plot 2D trajectory for the two provided state variables', default=False)


    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()    

    try:
        args.adaptive_rate = list([int(r) for r in args.adaptive_rate])
    except:
        print('Could not convert strings to integers for adaptive rate arg.')

    try:
        args.R_size = list([int(r) for r in args.R_size])
    except:
        print('Could not convert strings to integers for partition size arg.')

    try:
        args.R_width = list([float(r) for r in args.R_width])
    except:
        print('Could not convert strings to integers for partition width arg.')

    try:
        args.noise_strength_w = list([float(r) for r in args.noise_strength_w])
    except:
        print('Could not convert strings to floats for process noise strength')

    try:
        args.noise_strength_v = list([float(r) for r in args.noise_strength_v])
    except:
        print('Could not convert strings to floats for measurement noise strength')

    if args.plot_heatmap is not False:
        try:
            args.plot_heatmap = list([int(r) for r in args.plot_heatmap])
        except:
            print('Could not convert strings to integers for plot_heatmap arg.')

    if args.plot_trajectory_2D is not False:
        try:
            args.plot_trajectory_2D = list([int(r) for r in args.plot_trajectory_2D])
        except:
            print('Could not convert strings to integers for plot_trajectory_2D arg.')
    
    return args

def user_choice(title, items):
    '''
    Lets a user choose between the entries of `items`

    Parameters
    ----------
    title : str
        Title which to show above the choice.
    items : list
        List of strings for the items to choose from.

    Returns
    -------
    choice : str
        String of the chosen item.
    choice_id : int
        Integer index of the chosen item.

    '''
    
    if items == 'integer':
        # Let the user fill in an integer value of chocie
        
        print('\nMake a choice for:',str(title))
        print( ' -- Type an integer value above zero')
        choice = -1
        while type(choice) is not int or not choice > 0:
            choice = int(input('Please type your choice: '))
            
        print('\n >> Choice provided is',choice,'\n')
        
        choice_id = None
        
    elif len(items) > 1:
        # List of items provided, so let user choose
    
        print('\nMake a choice for:',str(title))
        for i,item in enumerate(items):
            print(' -- Type',str(i),'for',str(item))
        output = -1
        while output not in range(len(items)):
            output = int(input('Please choose your option: '))
          
        choice = items[output]
        print('\n >> Choice provided is',output,'('+str(choice)+')\n')
        
        choice_id = output
        
    else:
        # No items provided
        
        print('\nNo items provided; return output of zero')
        
        choice = 0
        choice_id = 0
            
    return choice, choice_id

def load_PRISM_result_file(mode_prefix, output_folder, model_name, 
                           N=0):
    '''
    Lets the user load results from a set of existing PRISM files

    Parameters
    ----------
    output_folder : str
        Location of the output folder.
    model_name : str
        Name of the model to load results from.
    N : TYPE
        Number of samples used in the model to load results from.

    Returns
    -------
    folder_to_load : str
        Folder where the files to load are located.
    policy_file : str
        Filename of the optimal policy results file.
    vector_file : str
        Filename of the optimal reward results file.

    '''

    suitable_folders = []

    # Select a suitable results file to load
    folder_list = next(os.walk(output_folder))
      
    if N != 0:
        # If N is not zero, then scenario-based
                 
        for folder in folder_list[1]:
            if folder.startswith(mode_prefix+'_'+model_name):
            
                run_folder = output_folder + folder
                subfolder_list = next(os.walk(run_folder))
                
                for subfolder in subfolder_list[1]:
                    if subfolder == 'N='+str(N):
                        suitable_folders += [subfolder_list[0] + '/' + subfolder]
    
        suitable_folders_trim = [None for i in range(len(suitable_folders))]
        
        for i,folder in enumerate(suitable_folders):
            # Remove prefix of the folder
            suitable_folders_trim[i] = '/'.join( folder.split('/')[-2:] )
            
    else:
        # If N is zero, then filter-based
        for folder in folder_list[1]:
            if folder.startswith(mode_prefix+'_'+model_name):
                
                run_folder = output_folder + folder
                subfolder_list = next(os.walk(run_folder))
                
                suitable_folders += [subfolder_list[0]]
        
        suitable_folders_trim = [None for i in range(len(suitable_folders))]
        
        for i,folder in enumerate(suitable_folders):
            # Remove prefix of the folder
            suitable_folders_trim[i] = '/'.join( folder.split('/')[-2:] )
        
    # If TRUE monte carlo simulations are performed
    _, folder_idx = user_choice( \
        'Choose a folder to load the PRISM results from...', suitable_folders_trim)
        
    folder_to_load = suitable_folders[folder_idx] + '/'
    
    #os.chdir(folder_to_load)
    
    policy_files = glob.glob(folder_to_load+"*policy.csv")
    vector_files = glob.glob(folder_to_load+"*vector.csv")
    
    if len(policy_files) > 0 and len(vector_files) > 0:
        
        policy_file = folder_to_load + policy_files[0]
        vector_file = folder_to_load + vector_files[0]
    
    else:
        
        policy_file = None
        vector_file = None
        
    return folder_to_load, policy_file, vector_file