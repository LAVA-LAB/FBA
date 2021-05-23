#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

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

def load_PRISM_result_file(output_folder, model_name, N):

    suitable_folders = []

    # Select a suitable results file to load
    folder_list = next(os.walk(output_folder))
                       
    for folder in folder_list[1]:
        if folder.startswith('ScAb_'+model_name):
        
            run_folder = output_folder + folder
            subfolder_list = next(os.walk(run_folder))
            
            for subfolder in subfolder_list[1]:
                if subfolder == 'N='+str(N):
                    suitable_folders += [subfolder_list[0] + '/' + subfolder]

    suitable_folders_trim = [None for i in range(len(suitable_folders))]
    for i,folder in enumerate(suitable_folders):
        # Remove prefix of the folder
        suitable_folders_trim[i] = '/'.join( folder.split('/')[-2:] )
        
    # If TRUE monte carlo simulations are performed
    _, folder_idx = user_choice( \
        'Choose a folder to load the PRISM results from...', suitable_folders_trim)
        
    folder_to_load = suitable_folders[folder_idx] + '/'
    os.chdir(folder_to_load)
    
    policy_files = glob.glob("*policy.csv")
    vector_files = glob.glob("*vector.csv")
    
    if len(policy_files) > 0 and len(vector_files) > 0:
        
        policy_file = folder_to_load + policy_files[0]
        vector_file = folder_to_load + vector_files[0]
    
    else:
        
        policy_file = None
        vector_file = None
        
    return folder_to_load, policy_file, vector_file