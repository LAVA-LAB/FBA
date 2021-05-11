#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("pgf")
import matplotlib.pyplot as plt

from ..commons import cm2inch

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def analyzeExperiment(M, experiment, setup):
    '''
    Analyze the experiments consisting of instances in `M`

    Parameters
    ----------
    M : dict
        Dictionary of abstraction objects for all instances.
    experiment : dict
        Experiment dictionary.
    analyzeExperiment

    Returns
    -------
    experiment : dict
        Experiment dictionary, with added results from the analysis.

    '''
    
    ######################
    # Plot 2D probability plot over time
    
    if 'Python' in setup.mdp['solver']:
        fig = plt.figure(figsize=cm2inch(12.2, 6))
        ax = plt.gca()
        
        # Plot probability reachabilities
        color = next(ax._get_lines.prop_cycler)['color']
        
        Nstart = 0
        nr_regions = M[0].abstr['nr_regions']
        
        experiment['prob']   = pd.DataFrame( index=range(nr_regions) )
        experiment['empi']   = pd.DataFrame( index=range(nr_regions) )
        
        for i in experiment['instances']:
        
            results = M[i].results
            mc = M[i].mc
            
            experiment['prob'] = experiment['prob'].join(pd.DataFrame(results['optimal_reward'][Nstart,:], columns=[i], index=range(M[i].abstr['nr_regions'])) )
            
            plt.plot(results['optimal_reward'][Nstart,:], label='case '+str(i), linewidth=1, color=color)
            if setup.montecarlo['enabled']:
                
                experiment['empi'] = experiment['empi'].join( pd.DataFrame(mc['results']['reachability_probability'][:,0], columns=[i], index=range(M[i].abstr['nr_regions'])) )
                
                plt.plot(mc['results']['reachability_probability'][:,0],
                         linewidth=1, color=color, linestyle='dashed')
        
            color = next(ax._get_lines.prop_cycler)['color']
        
        # Styling plot
        plt.xlabel('States')
        plt.ylabel('Reachability probability')
        plt.legend(loc="upper left")
        
        # Set axis limits
        plt.xlim( 0, nr_regions )
        
        # Set tight layout
        fig.tight_layout()
                    
        # Save figure
        filename = setup.directories['outputF']+'reachability_probability_all_instances'
        plt.savefig(filename+'.pdf', bbox_inches='tight')
        plt.savefig(filename+'.png', bbox_inches='tight')
    
    ######################
    
    # Store timer results to Excel
    output_file = setup.directories['outputF']+'data_export.xlsx'
       
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    # Write dataframes to a different worksheet
    experiment['time'].to_excel(writer, sheet_name='Time')
    
    if 'Python' in setup.mdp['solver']:
        experiment['prob'].to_excel(writer, sheet_name='Prob. reach.')
        if setup.montecarlo['enabled']:
            experiment['empi'].to_excel(writer, sheet_name='Empirical reach.')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    ######################
    
    return experiment