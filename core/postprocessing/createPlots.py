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

import numpy as np              # Import Numpy for computations
import pandas as pd             # Import Pandas to store data in frames
import matplotlib.pyplot as plt # Import Pyplot to generate plots

# Load main classes and methods
from matplotlib import cm
from scipy.spatial import ConvexHull
from matplotlib.patches import Rectangle

from ..commons import printWarning, mat_to_vec, cm2inch


def createPartitionPlot(i_tup, j_tup, j, delta_plot, setup, model, \
                        abstr, allVerticesNested, predecessor_set):
    
    '''
    
    Create partition plot for the current abstraction instance.

    Parameters
    ----------
    i_tup : tuple
        Tuple of indices to plot against
    j_tup : tuple
        Tuple of indices to create a cross-section for (i.e. are fixed)
    j : int
        Index of the center region of the state space partition.
    delta_plot : int
        Value for which to show the set controllable to `d_j` in `delta` steps.
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    allVerticesNested : list
        Nested lists containing all origin points of all regions.
    predecessor_set : array
        Vertices of the predecessor set

    Returns
    -------
    None.
    '''

    
    i0,i1 = i_tup
    j0,j1 = j_tup
    
    fig = plt.figure(figsize=cm2inch(12, 7))
    ax = fig.add_subplot(111)
    
    # If number of dimensions is 2, create 2D plot
    if model.n <= 2:        
        plt.xlabel('$x_1$', labelpad=0)
        plt.ylabel('$x_2$', labelpad=-10)
    else:
        plt.xlabel('$x_'+str(i0)+'$', labelpad=0)
        plt.ylabel('$x_'+str(i1)+'$', labelpad=-10)
    
    for k,poly in enumerate(allVerticesNested):

        if model.n <= 2 or ( \
            abstr['P'][k]['center'][j0] == model.setup['partition']['origin'][j0] and \
            abstr['P'][k]['center'][j1] == model.setup['partition']['origin'][j1]):

            # Convert poly list to numpy array            
            polyMat = np.array(poly)
            
            # Plot partitions and label
            ax.text(abstr['P'][k]['center'][i0], abstr['P'][k]['center'][i1], k, \
                      verticalalignment='center', horizontalalignment='center' )  
            hull = ConvexHull(polyMat, qhull_options='QJ')
            ax.plot(polyMat[hull.vertices,i0], polyMat[hull.vertices,i1], lw=1)
            ax.plot([polyMat[hull.vertices[0],i0], polyMat[hull.vertices[-1],i0]], \
                      [polyMat[hull.vertices[0],i1], polyMat[hull.vertices[-1],i1]], lw=1)
        
    ax.plot(0.3946, 4.5967, 'ro', lw=1)
    ax.plot(-0.2611, 4.1995, 'ro', lw=1)
    ax.plot(-1.7577, 3.2995, 'bo', lw=1)
    ax.plot(-2.1345, 2.6954, 'go', lw=1)
        
    for k,target_point in enumerate(abstr['target']['d']):
          
        if model.n <= 2 or ( \
            abstr['P'][k]['center'][j0] == model.setup['partition']['origin'][j0] and \
            abstr['P'][k]['center'][j1] == model.setup['partition']['origin'][j1]):
        
            # Plot target point
            plt.scatter(target_point[i0], target_point[i1], c='k', s=6)
    
    if setup.plotting['partitionPlot_plotHull']:
        # Plot convex hull of the preimage of the target point
        hull = ConvexHull(predecessor_set, qhull_options='QJ')
        ax.plot(predecessor_set[hull.vertices,i0], predecessor_set[hull.vertices,i1], 'ro--', lw=1) #2
        ax.plot([predecessor_set[hull.vertices[0],i0], predecessor_set[hull.vertices[-1],i0]], \
                  [predecessor_set[hull.vertices[0],i1], predecessor_set[hull.vertices[-1],i1]], 'ro--', lw=1) #2
        
    # Set tight layout
    fig.tight_layout()
                
    # Save figure
    filename = setup.directories['outputF']+'partitioning_'+str(delta_plot)+'_coords=('+str(i0)+','+str(i1)+')'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
    # If number of dimensions is 3, create 3D plot
    if model.n == 3:
        
        fig = plt.figure(figsize=cm2inch(5.33, 4))
        ax = fig.add_subplot(111, projection="3d")
        
        for k,poly in enumerate(allVerticesNested):
    
            # Convert list to numpy array
            poly = np.array(poly)
            
            # Create hull for this partition
            hull = ConvexHull(poly)
            
            # Plot defining corner points
            for t in range(2**model.n):
                # If contained in the inverse image, plot green. blue otherwise
                '''
                if enabledPolypoints[j][k, t]:
                    color = "g"
                else:
                    color = "b"
                '''
                color = "b"
                ax.scatter(poly[t,0], poly[t,1], poly[t,2], c=color)
            
        # Make axis label
        for i in ["x", "y", "z"]:
            eval("ax.set_{:s}label('{:s}')".format(i, i))
    
        # Set tight layout
        fig.tight_layout()
                    
        # Save figure
        filename = setup.directories['outputF']+'partitioning'
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    else:
        print('Number of dimensions is larger than 3, so partition plot omitted')

def set_axes_equal(ax: plt.Axes):
    """
    Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)
    
def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    # ax.set_zlim3d([z - radius, z + radius])

def createProbabilityPlots(setup, plot, N, model, results, abstr, mc):
    '''
    Create the result plots for the abstraction instance.

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    plot : dict
        Dictionary containing info about plot settings
    N : int
        Finite time horizon.
    model : dict
        Main dictionary of the LTI system model.
    results : dict
        Dictionary containing all results from solving the MDP.
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    mc : dict
        Dictionary containing all data relevant to the Monte Carlo simulations.

    Returns
    -------
    None.

    '''
    
    # Plot 2D probability plot over time
    
    if N/2 != round(N/2):
        printWarning('WARNING: '+str(N/2)+' is no valid integer index')
        printWarning('Print results for time index k='+str(int(np.floor(N/2)-1))+' instead')
    
    if setup.montecarlo['enabled']:
        fig = plt.figure(figsize=cm2inch(14, 7))
    else:
        fig = plt.figure(figsize=cm2inch(8, 6))
    ax = plt.gca()
    
    if 'start' in plot['N']:
        # Plot probability reachabilities
        color = next(ax._get_lines.prop_cycler)['color']
        
        plt.plot(results['optimal_reward'][plot['N']['start'],:], label='k='+str(plot['T']['start']), linewidth=1, color=color)
        if setup.montecarlo['enabled'] and not setup.montecarlo['init_states']:
            plt.plot(mc['results']['reachability_probability'][:,0], label='Monte carlo (k='+str(plot['T']['start'])+')', \
                     linewidth=1, color=color, linestyle='dashed')
    
    if 'half' in plot['N']:
        color = next(ax._get_lines.prop_cycler)['color']
        
        plt.plot(results['optimal_reward'][plot['N']['half'],:], label='k='+str(plot['T']['half']), linewidth=1, color=color)
        if setup.montecarlo['enabled'] and not setup.montecarlo['init_states']:
            plt.plot(mc['results']['reachability_probability'][:,1], label='Monte carlo (k='+str(plot['T']['half'])+')', \
                     linewidth=1, color=color, linestyle='dashed')
    
    if 'final' in plot['N']:
        color = next(ax._get_lines.prop_cycler)['color']
        
        plt.plot(results['optimal_reward'][plot['N']['final'],:], label='k='+str(plot['T']['final']), linewidth=1, color=color)
        if setup.montecarlo['enabled'] and not setup.montecarlo['init_states']:
            plt.plot(mc['results']['reachability_probability'][:,2], label='Monte carlo (k='+str(plot['T']['final'])+')', \
                     linewidth=1, color=color, linestyle='dashed')
    
    # Styling plot
    plt.xlabel('States')
    plt.ylabel('Reachability probability')
    plt.legend(loc="upper left")
    
    # Set tight layout
    fig.tight_layout()
                
    # Save figure
    filename = setup.directories['outputFcase']+'reachability_probability'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
    ######################
    # Determine dimension of model
    m = model.setup['partition']['nrPerDim']
    
    # Plot 3D probability plot for selected time steps
    if model.n > 2:
        printWarning('Number of dimensions is larger than 2, so 3D reachability plot omitted')
    
    else:
        # Plot 3D probability results
        plot3D      = dict()
        
        # Retreive X and Y values of the centers of the regions
        plot3D['x'] = np.zeros(len(abstr['P']))
        plot3D['y'] = np.zeros(len(abstr['P']))
        
        for i in range(len(abstr['P'])):
            plot3D['x'][i]  = abstr['P'][i]['center'][0]
            plot3D['y'][i]  = abstr['P'][i]['center'][1]
        
        plot3D['x'] = np.reshape(plot3D['x'], (m[0],m[1]))
        plot3D['y'] = np.reshape(plot3D['y'], (m[0],m[1]))
        
        fig_comb    = plt.figure(figsize=cm2inch(16,5.33))
        fig_ind     = 0
        ax_comb     = dict()
        surf_comb   = dict()
        
        time_list = [ [plot['N'][key], plot['T'][key]] for key in plot['N'].keys() ]
        
        for [k,t] in time_list:
            # Create figure
            fig = plt.figure(figsize=cm2inch(8,5.33))
            ax  = plt.axes(projection='3d')
            
            # Increase combo figure index
            fig_ind += 1
            
            # Create combo subplot
            ax_comb[fig_ind]  = fig_comb.add_subplot(1,3,fig_ind, projection="3d")

            # Determine matrix of probability values
            Z   = np.reshape(results['optimal_reward'][k,:], (m[0],m[1]))
            
            # Plot the surface
            surf = ax.plot_surface(plot3D['x'], plot3D['y'], Z, 
                            cmap=cm.coolwarm, linewidth=0, antialiased=False)
            
            # Add subfigure in combined figure
            surf_comb[fig_ind] = ax_comb[fig_ind].plot_surface(plot3D['x'], plot3D['y'], Z, 
                            cmap=cm.coolwarm, linewidth=0, antialiased=False)
            
            # Customize the z axis
            ax.set_zlim(0,1)
            
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)
            
            # Set title and axis format
            ax.title.set_text('Reachability probability at time k = '+str(t))
            x_row = mat_to_vec( plot3D['x'] )
            y_col = mat_to_vec( plot3D['y'] )
            n_ticks = 5
            plt.xticks(np.arange(min(x_row), max(x_row)+1, \
                                 (max(x_row) - min(x_row))/n_ticks ))
            plt.yticks(np.arange(min(y_col), max(y_col)+1, \
                                 (max(y_col) - min(y_col))/n_ticks ))
            plt.tick_params(pad=-3)
                
            plt.xlabel('x_1', labelpad=-6)
            plt.ylabel('x_2', labelpad=-6)
            
            # Set tight layout
            fig.tight_layout()
            
            # Save figure
            filename = setup.directories['outputFcase']+'3d_reachability_k='+str(t)
            for form in setup.plotting['exportFormats']:
                plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
            
            # Style combined subplot figure
            ax_comb[fig_ind].set_zlim(0,1)
            ax_comb[fig_ind].set_title('time k = '+str(t))
            ax_comb[fig_ind].set_xticks(np.arange(min(x_row), max(x_row)+1, \
                                 (max(x_row) - min(x_row))/n_ticks ))
            ax_comb[fig_ind].set_yticks(np.arange(min(y_col), max(y_col)+1, \
                                 (max(y_col) - min(y_col))/n_ticks ))
            ax_comb[fig_ind].tick_params(pad=-3)
            ax_comb[fig_ind].set_xlabel('x_1', labelpad=-6)
            ax_comb[fig_ind].set_ylabel('x_2', labelpad=-6)
            
            '''
            ax.set_box_aspect([1,1,1])
            set_axes_equal(ax)
            ax_comb[fig_ind].set_box_aspect([1,1,1])
            set_axes_equal(ax_comb[fig_ind])
            '''
        
        # Style combined subplot figure
        fig_comb.tight_layout()
        
        # Save figure
        filename = setup.directories['outputFcase']+'3d_reachability_all'
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
    ######################
    # Visualize matrix of action types, based on nr of time steps grouped

    fig, ax = plt.subplots(figsize=cm2inch(16,6))
    
    # Shortcut to data
    data = results['optimal_delta']
    
    #get discrete colormap
    cmap = plt.get_cmap('Greys', np.max(data)-np.min(data)+1)
    
    # set limits .5 outside true range
    mat = ax.matshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)
    
    #tell the colorbar to tick at integers
    plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1), shrink=0.5, aspect=5)
    
    plt.xlabel('State')
    plt.ylabel('Time step ($k$)')
    
     # Set tight layout
    fig.tight_layout()
                
    # Save figure
    filename = setup.directories['outputFcase']+'policy_action_delta_value'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
def policyPlot(setup, model, results, abstr):
    '''
    Create the policy plot for the current abstraction instance

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    results : dict
        Dictionary containing all results from solving the MDP.
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.

    Returns
    -------
    None.

    '''
    
    ix = 0
    iy = 1
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('Zone temp.', labelpad=0)
    plt.ylabel('Radiator temp.', labelpad=0)

    width = np.array(model.setup['partition']['width'])
    domainMax = width * np.array(model.setup['partition']['nrPerDim']) / 2
    
    min_xy = model.setup['partition']['origin'] - domainMax
    max_xy = model.setup['partition']['origin'] + domainMax
    
    major_ticks_x = np.arange(min_xy[ix]+1, max_xy[ix]+1, 4*width[ix])
    major_ticks_y = np.arange(min_xy[iy]+1, max_xy[iy]+1, 4*width[iy])
    
    minor_ticks_x = np.arange(min_xy[ix], max_xy[ix]+1, width[ix])
    minor_ticks_y = np.arange(min_xy[iy], max_xy[iy]+1, width[iy])
    
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    
    # plt.grid(which='major', color='#CCCCCC', linewidth=0.3)
    plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
    
    # Goal x-y limits
    ax.set_xlim(min_xy[ix], max_xy[ix])
    ax.set_ylim(min_xy[iy], max_xy[iy])
    
    # Draw goal states
    for goal in abstr['goal']:
        
        goalState = abstr['P'][goal]
        
        goal_lower = [goalState['low'][ix], goalState['low'][iy]]
        goalState = Rectangle(goal_lower, width=width[ix], height=width[iy], color="green", alpha=0.3, linewidth=None)
        ax.add_patch(goalState)
    
    # Draw critical states
    for crit in abstr['critical']:
        
        critState = abstr['P'][crit]
        
        critStateLow = [critState['low'][ix], critState['low'][iy]]
        criticalState = Rectangle(critStateLow, width=width[ix], height=width[iy], color="red", alpha=0.3, linewidth=None)
        ax.add_patch(criticalState)
    
    frequency = 1
    
    for state_from, action in enumerate(results['optimal_policy'][0,:]):
        if state_from % frequency == 0 and action != -1:
            
            center_from = abstr['P'][state_from]['center']
            center_to   = abstr['target']['d'][int(action)]
            
            dxy = center_to - center_from
            
            # Draw arrow
            ax.arrow(center_from[0],center_from[1],dxy[0],dxy[1], 
                    lw=0.1, head_width=0.1, head_length=0.1, 
                    length_includes_head=True, edgecolor='r') #, edgecolor=edgecol, facecolor=facecol, alpha=alpha)
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'policy'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()
        
def UAVplots(ScAb, case_id, writer):
    '''
    Create the trajectory plots for the UAV benchmarks

    Parameters
    ----------
    ScAb : abstraction instance
        Full object of the abstraction being plotted for
    case_id : int
        Index for the current abstraction iteration
    writer : XlsxWriter
        Writer object to write results to Excel

    Returns
    -------
    performance_df : Pandas DataFrame
        DataFrame containing the empirical performance results

    '''
    
    from core.mainFunctions import computeRegionCenters
    from core.commons import setStateBlock
    
    # Determine desired state IDs
    if ScAb.basemodel.name == 'UAV':
        if ScAb.basemodel.modelDim == 2:
            x_init = setStateBlock(ScAb.basemodel.setup['partition'], a=[-6], b=[0], c=[-6], d=[0])
            
            cut_value = np.zeros(2)
            for i,d in enumerate(range(1, ScAb.basemodel.n, 2)):
                if ScAb.basemodel.setup['partition']['nrPerDim'][d]/2 != round( ScAb.basemodel.setup['partition']['nrPerDim'][d]/2 ):
                    cut_value[i] = 0
                else:
                    cut_value[i] = ScAb.basemodel.setup['partition']['width'][d] / 2                
            
        elif ScAb.basemodel.modelDim == 3:
            x_init = setStateBlock(ScAb.basemodel.setup['partition'], a=[-6], b=[0], c=[6], d=[0], e=[-6], f=[0])
            
            cut_value = np.zeros(3)
            for i,d in enumerate(range(1, ScAb.basemodel.n, 2)):
                if ScAb.basemodel.setup['partition']['nrPerDim'][d]/2 != round( ScAb.basemodel.setup['partition']['nrPerDim'][d]/2 ):
                    cut_value[i] = 0
                else:
                    cut_value[i] = ScAb.basemodel.setup['partition']['width'][d] / 2          
            
    # Compute all centers of regions associated with points
    x_init_centers = computeRegionCenters(np.array(x_init), ScAb.basemodel.setup['partition'])
    
    # Filter to only keep unique centers
    x_init_unique = np.unique(x_init_centers, axis=0)
    
    state_idxs = [ScAb.abstr['allCenters'][tuple(c)] for c in x_init_unique 
                                   if tuple(c) in ScAb.abstr['allCenters']]
    
    print(' -- Perform simulations for initial states:',state_idxs)
    
    ScAb.setup.montecarlo['init_states'] = state_idxs
    ScAb.setup.montecarlo['iterations'] = 10000
    ScAb.monteCarlo()
    
    PRISM_reach = ScAb.results['optimal_reward'][0,state_idxs]
    empirical_reach = ScAb.mc['results']['reachability_probability'][state_idxs]
    
    print('Probabilistic reachability (PRISM): ',PRISM_reach)
    print('Empirical reachability (Monte Carlo):',empirical_reach)
    
    performance_df = pd.DataFrame( {'PRISM reachability': PRISM_reach.flatten(),
                                    'Empirical reachability': empirical_reach.flatten() }, index=[case_id] )
    performance_df.to_excel(writer, sheet_name='Performance')
    
    itersToShow = 10
    
    traces = []
    for i in state_idxs:
        for j in range(itersToShow):
            traces += [ScAb.mc['traces'][0][i][j]]
    
    min_delta = int(min(ScAb.setup.deltas))
    
    if ScAb.basemodel.modelDim == 2:
        UAVplot2D( ScAb.setup, ScAb.model[min_delta], ScAb.abstr, traces, cut_value )
    
    elif ScAb.basemodel.modelDim == 3:
        if ScAb.setup.main['iterative'] is False or ScAb.setup.plotting['3D_UAV']:
        
            # Only plot trajectory plot in non-iterative mode (because it pauses the script)
            UAVplot3d_visvis( ScAb.setup, ScAb.model[min_delta], ScAb.abstr, traces, cut_value ) 
    
    return performance_df
    
def UAVplot2D(setup, model, abstr, traces, cut_value):
    '''
    Create 2D trajectory plots for the 2D UAV benchmark

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    traces : list
        Nested list containing the trajectories (traces) to plot for
    cut_value : array
        Values to create the cross-section for

    Returns
    -------
    None.

    '''
    
    from scipy.interpolate import interp1d
    
    ix = 0
    iy = 2
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(model.setup['partition']['width'])
    domainMax = width * np.array(model.setup['partition']['nrPerDim']) / 2
    
    min_xy = model.setup['partition']['origin'] - domainMax
    max_xy = model.setup['partition']['origin'] + domainMax
    
    major_ticks_x = np.arange(min_xy[ix]+1, max_xy[ix]+1, 4*width[ix])
    major_ticks_y = np.arange(min_xy[iy]+1, max_xy[iy]+1, 4*width[iy])
    
    minor_ticks_x = np.arange(min_xy[ix], max_xy[ix]+1, width[ix])
    minor_ticks_y = np.arange(min_xy[iy], max_xy[iy]+1, width[iy])
    
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    
    # plt.grid(which='major', color='#CCCCCC', linewidth=0.3)
    plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
    
    # Goal x-y limits
    ax.set_xlim(min_xy[ix], max_xy[ix])
    ax.set_ylim(min_xy[iy], max_xy[iy])
    
    # Draw goal states
    for goal in abstr['goal']:
        
        goalState = abstr['P'][goal]
        if goalState['center'][1] == cut_value[0] and goalState['center'][3] == cut_value[1]:
        
            goal_lower = [goalState['low'][ix], goalState['low'][iy]]
            goalState = Rectangle(goal_lower, width=width[ix], height=width[iy], color="green", alpha=0.3, linewidth=None)
            ax.add_patch(goalState)
    
    # Draw critical states
    for crit in abstr['critical']:
        
        critState = abstr['P'][crit]
        if critState['center'][1] == cut_value[0] and critState['center'][3] == cut_value[1]:
        
            critStateLow = [critState['low'][ix], critState['low'][iy]]
            criticalState = Rectangle(critStateLow, width=width[ix], height=width[iy], color="red", alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)
    
    with plt.rc_context({"font.size": 5}):        
        # Draw every X-th label
        skip = 1
        for i in range(0, abstr['nr_regions'], skip):
            
            state = abstr['P'][i]
            if state['center'][1] == cut_value[0] and state['center'][3] == cut_value[1]:
                            
                ax.text(abstr['P'][i]['center'][ix], abstr['P'][i]['center'][iy], i, \
                          verticalalignment='center', horizontalalignment='center' ) 
            
    # Add traces
    for i,trace in enumerate(traces):
        
        if len(trace) < 3:
            printWarning('Warning: trace '+str(i)+' has length of '+str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, ix]
        y = trace_array[:, iy]
        points = np.array([x,y]).T
        
        # Plot precise points
        plt.plot(*points.T, 'o', markersize=4, color="black");
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, 75)
        
        interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
        interpolated_points = interpolator(alpha)
        
        # Plot trace
        plt.plot(*interpolated_points.T, '-', color="blue", linewidth=1);
        # plt.plot(x_values, y_values, color="blue")
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'drone_trajectory'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()
    
def UAVplot3d_visvis(setup, model, abstr, traces, cut_value):
    '''
    Create 3D trajectory plots for the 3D UAV benchmark

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    traces : list
        Nested list containing the trajectories (traces) to plot for
    cut_value : array
        Values to create the cross-section for

    Returns
    -------
    None.

    '''
    
    from scipy.interpolate import interp1d
    import visvis as vv
    
    fig = vv.figure()
    f = vv.clf()
    a = vv.cla()
    fig = vv.gcf()
    ax = vv.gca()
    
    ix = 0
    iy = 2
    iz = 4
    
    regionWidth_xyz = np.array([model.setup['partition']['width'][0], 
                                model.setup['partition']['width'][2], 
                                model.setup['partition']['width'][4]])    
    
    # Draw goal states
    for goal in abstr['goal']:
        
        goalState = abstr['P'][goal]
        if goalState['center'][1] == cut_value[0] and goalState['center'][3] == cut_value[1] and goalState['center'][5] == cut_value[2]:
        
            center_xyz = np.array([goalState['center'][0], 
                                   goalState['center'][2], 
                                   goalState['center'][4]])
            
            goal = vv.solidBox(tuple(center_xyz), scaling=tuple(regionWidth_xyz))
            goal.faceColor = (0,1,0,0.5)
            
    # Draw critical states
    for crit in abstr['critical']:
        
        critState = abstr['P'][crit]
        if critState['center'][1] == cut_value[0] and critState['center'][3] == cut_value[1] and critState['center'][5] == cut_value[2]:
        
            center_xyz = np.array([critState['center'][0], 
                                   critState['center'][2], 
                                   critState['center'][4]])    
        
            critical = vv.solidBox(tuple(center_xyz), scaling=tuple(regionWidth_xyz))
            critical.faceColor = (1,0,0,0.5)
    
    # Add traces
    for i,trace in enumerate(traces):
        
        if len(trace) < 3:
            printWarning('Warning: trace '+str(i)+' has length of '+str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        # Extract x,y coordinates of trace
        x = trace_array[:, ix]
        y = trace_array[:, iy]
        z = trace_array[:, iz]
        points = np.array([x,y,z]).T
        
        # Plot precise points
        vv.plot(x,y,z, lw=0, mc='b', ms='.')
        
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, 75)
        
        interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
        interpolated_points = interpolator(alpha)
        
        xp = interpolated_points[:,0]
        yp = interpolated_points[:,1]
        zp = interpolated_points[:,2]
        
        # Plot trace
        vv.plot(xp,yp,zp, lw=1, lc='b')
    
    ax.axis.xLabel = 'X'
    ax.axis.yLabel = 'Y'
    ax.axis.zLabel = 'Z'
    
    app = vv.use()
    f.relativeFontSize = 1.6
    # ax.position.Correct(dh=-5)
    vv.axis('tight', axes=ax)
    fig.position.w = 700
    fig.position.h = 600
    im = vv.getframe(vv.gcf())
    ax.SetView({'zoom':0.042, 'elevation':25, 'azimuth':-35})
    
    filename = setup.directories['outputFcase'] + 'UAV_paths_screenshot.png'
    
    vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())
    app.Run()
    
def reachabilityHeatMap(ScAb):
    '''
    Create heat map for the BAS benchmarks

    Parameters
    ----------
    ScAb : abstraction instance
        Full object of the abstraction being plotted for

    Returns
    -------
    None.

    '''
    
    import seaborn as sns
    from ..mainFunctions import definePartitions

    if ScAb.basemodel.name == 'building_2room':
    
        x_nr = ScAb.basemodel.setup['partition']['nrPerDim'][0]
        y_nr = ScAb.basemodel.setup['partition']['nrPerDim'][1]
        
        cut_centers = definePartitions(ScAb.basemodel.n, [x_nr, y_nr, 1, 1], 
               ScAb.basemodel.setup['partition']['width'], 
               ScAb.basemodel.setup['partition']['origin'], onlyCenter=True)
        
    elif ScAb.basemodel.name == 'building_1room':
    
        x_nr = ScAb.basemodel.setup['partition']['nrPerDim'][0]
        y_nr = ScAb.basemodel.setup['partition']['nrPerDim'][1]
        
        cut_centers = definePartitions(ScAb.basemodel.n, [x_nr, y_nr], 
               ScAb.basemodel.setup['partition']['width'], 
               ScAb.basemodel.setup['partition']['origin'], onlyCenter=True)
        
    elif ScAb.basemodel.name == 'UAV':
        
        x_nr = ScAb.basemodel.setup['partition']['nrPerDim'][0]
        y_nr = ScAb.basemodel.setup['partition']['nrPerDim'][2]
        
        cut_centers = definePartitions(ScAb.basemodel.n, [x_nr, 1, y_nr, 1], 
               ScAb.basemodel.setup['partition']['width'], 
               ScAb.basemodel.setup['partition']['origin'], onlyCenter=True)
                          
    cut_values = np.zeros((x_nr, y_nr))
    cut_coords = np.zeros((x_nr, y_nr, ScAb.basemodel.n))
    
    cut_idxs = [ScAb.abstr['allCenters'][tuple(c)] for c in cut_centers 
                                   if tuple(c) in ScAb.abstr['allCenters']]              
    
    for i,(idx,center) in enumerate(zip(cut_idxs, cut_centers)):
        
        j = i % y_nr
        k = i // y_nr
        
        cut_values[k,j] = ScAb.results['optimal_reward'][0,idx]
        cut_coords[k,j,:] = center
    
    cut_df = pd.DataFrame( cut_values, index=cut_coords[:,0,0], columns=cut_coords[0,:,1] )
    
    fig = plt.figure(figsize=cm2inch(9, 8))
    ax = sns.heatmap(cut_df.T, cmap="jet", #YlGnBu
             vmin=0, vmax=1)
    ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.invert_yaxis()
    
    # xticks = [t if i % 5 == 0 else '' for i,t in enumerate(cut_df.index.values.round(1))]
    # yticks = [t if i % 5 == 0 else '' for i,t in enumerate(cut_df.columns.values.round(1))]
    
    # ax.set_xticklabels(xticks, size = 13)
    # ax.set_yticklabels(yticks, size = 13)
    ax.set_xlabel('Temp. zone 1', fontsize=15)
    ax.set_ylabel('Temp. zone 2', fontsize=15)
    ax.set_title("N = "+str(ScAb.setup.scenarios['samples']),fontsize=20)
    
    # Set tight layout
    fig.tight_layout()

    # Save figure
    filename = ScAb.setup.directories['outputFcase']+'safeset_N='+str(ScAb.setup.scenarios['samples'])
    for form in ScAb.setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()