#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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