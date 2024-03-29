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

import numpy as np              # Import Numpy for computations
import pandas as pd             # Import Pandas to store data in frames
import matplotlib.pyplot as plt # Import Pyplot to generate plots
import cv2

plt.ioff()

from core.mainFunctions import definePartitions
import seaborn as sns

# Load main classes and methods
from matplotlib import cm
from scipy.spatial import ConvexHull
from matplotlib.patches import Rectangle

from core.mainFunctions import computeRegionCenters

from ..commons import setStateBlock, printWarning, mat_to_vec, cm2inch, confidence_ellipse

def partitionPlot2D(i_tup, j_tup, j, delta_plot, setup, model, \
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
    
    fig = plt.figure(figsize=cm2inch(12, 7))
    ax = fig.add_subplot(111)
    
    # If number of dimensions is 2, create 2D plot
    if model['n'] <= 2:
        plt.xlabel('$x_1$', labelpad=0)
        plt.ylabel('$x_2$', labelpad=-10)
        
        i0 = 0
        i1 = 1
        
        print(' -- Plot 2D partition plot')
        
    else:
        i0,i1 = i_tup
        j0,j1 = j_tup
        
        plt.xlabel('$x_'+str(i0)+'$', labelpad=0)
        plt.ylabel('$x_'+str(i1)+'$', labelpad=-10)
        
        print(' -- Plot partition plot w.r.t. axes',str(i_tup))
        
    for k,target_point in enumerate(abstr['target']['d']):
        
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
    
    # plt.show()
                
    # Save figure
    filename = setup.directories['outputF']+'partitioning_'+str(delta_plot)+'_coords=('+str(i0)+','+str(i1)+')'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
def partitionPlot3D(setup, model, allVerticesNested, predecessor_set):
    
    '''
    
    Create partition plot for the current abstraction instance.

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    allVerticesNested : list
        Nested lists containing all origin points of all regions.
    predecessor_set : array
        Vertices of the predecessor set

    Returns
    -------
    None.
    '''
        
    print(' -- Create 3D partition plot')
    
    fig = plt.figure(figsize=cm2inch(5.33, 4))
    ax = fig.add_subplot(111, projection="3d")
    
    for k,poly in enumerate(allVerticesNested):

        # Convert list to numpy array
        poly = np.array(poly)
        
        # Create hull for this partition
        hull = ConvexHull(poly)
        
        # Plot defining corner points
        for t in range(2**model['n']):
            # If contained in the inverse image, plot green. blue otherwise
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

def createProbabilityPlots(setup, modelDim, partition, mdp, abstr, mc):
    '''
    Create the result plots for the abstraction instance.

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    modelDim : int
        Dimension of the model.
    partition : dict
        Dictionary containing info about the partitioning
    mdp : dict
        Dictionary containing all iMDP data
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    mc : dict
        Dictionary containing all data relevant to the Monte Carlo simulations.

    Returns
    -------
    None.

    '''
    
    # Plot 2D probability plot
    
    if setup.montecarlo['enabled']:
        fig = plt.figure(figsize=cm2inch(14, 7))
    else:
        fig = plt.figure(figsize=cm2inch(8, 6))
    ax = plt.gca()
    
    # Plot probability reachabilities
    color = next(ax._get_lines.prop_cycler)['color']
    
    plt.plot(mdp.MAIN_DF['opt_reward'][: abstr['nr_regions']], 
             label='k=0', linewidth=1, color=color)
    if setup.montecarlo['enabled'] and not setup.montecarlo['init_states']:
        plt.plot(mc['reachability_probability'], label='Monte carlo (k=0)', \
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
    m = partition['nrPerDim']
    
    # Plot 3D probability plot for selected time steps
    if modelDim > 2:
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
        
        # Create figure for 2D probability
        fig = plt.figure(figsize=cm2inch(8,5.33))
        ax  = plt.axes(projection='3d')
        
        # Determine matrix of probability values
        Z   = np.reshape(mdp.MAIN_DF['opt_reward'][:abstr['nr_regions']].to_numpy(), 
                         (m[0],m[1]))
        
        # Plot the surface
        surf = ax.plot_surface(plot3D['x'], plot3D['y'], Z, 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Customize the z axis
        ax.set_zlim(0,1)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Set title and axis format
        ax.title.set_text('Reachability probability at time k=0')
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
        filename = setup.directories['outputFcase']+'3d_reachability_k=0'
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
def policyPlot(setup, model, partition, results, abstr):
    '''
    Create the policy plot for the current abstraction instance

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    partition : dict
        Dictionary containing info about the partitioning
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

    width = np.array(partition['width'])
    domainMax = width * np.array(partition['nrPerDim']) / 2
    
    min_xy = partition['origin'] - domainMax
    max_xy = partition['origin'] + domainMax
    
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
    for goal in abstr['goal']['zero_bound']:
        
        goalState = abstr['P'][goal]
        
        goal_lower = [goalState['low'][ix], goalState['low'][iy]]
        goalState = Rectangle(goal_lower, width=width[ix], height=width[iy], color="green", alpha=0.3, linewidth=None)
        ax.add_patch(goalState)
    
    # Draw critical states
    for crit in abstr['critical']['zero_bound']:
        
        critState = abstr['P'][crit]
        
        critStateLow = [critState['low'][ix], critState['low'][iy]]
        criticalState = Rectangle(critStateLow, width=width[ix], height=width[iy], color="red", alpha=0.3, linewidth=None)
        ax.add_patch(criticalState)
    
    frequency = 1
    
    for state_from, action in enumerate(results['policy']['action'][1][0,:]):
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
        
    # plt.show()
        
def UAV_3D_plotLayout(Ab):
    '''
    Create a plot that shows the layout of the UAV problem
    '''
    
    cut_value = np.zeros(3)
    for i,d in enumerate(range(1, Ab.system.LTI['n'], 2)):
        if Ab.system.partition['nrPerDim'][d]/2 != round( Ab.system.partition['nrPerDim'][d]/2 ):
            cut_value[i] = 0
        else:
            cut_value[i] = Ab.system.partition['width'][d] / 2    
    
    UAVplot3d_visvis( Ab.setup, Ab.system.spec, cut_value, traces=[] ) 
    
def trajectoryPlot(Ab, case_id, writer = None):
    '''
    Create the trajectory plots for the UAV benchmarks

    Parameters
    ----------
    Ab : abstraction instance
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
    
    from ..filterBasedAbstraction import MonteCarloSim

    itersToShow = 1
    trajectoriesToCompute = 25

    # Determine desired state IDs
    if Ab.system.name == 'UAV_2D':
        x_init = setStateBlock(Ab.system.partition, a=[-8], b=[0], c=[-8], d=[0])
        
        itersToShow = 1
        
        cut_value = np.zeros(2)
        for i,d in enumerate(range(1, Ab.system.LTI['n'], 2)):
            if Ab.system.partition['nrPerDim'][d]/2 != round( Ab.system.partition['nrPerDim'][d]/2 ):
                cut_value[i] = 0
            else:
                cut_value[i] = Ab.system.partition['width'][d] / 2                

    elif Ab.system.name == 'UAV_3D':
            x_init = Ab.system.x0 #setStateBlock(Ab.system.partition, a=[-11], b=[0], c=[5], d=[0], e=[-5], f=[0])
            
            itersToShow = 10
            
            cut_value = np.zeros(3)
            for i,d in enumerate(range(1, Ab.system.LTI['n'], 2)):
                if Ab.system.partition['nrPerDim'][d]/2 != round( Ab.system.partition['nrPerDim'][d]/2 ):
                    cut_value[i] = 0
                else:
                    cut_value[i] = Ab.system.partition['width'][d] / 2          
                    
    elif Ab.system.name in ['shuttle', 'shuttleB']:
        x_init = setStateBlock(Ab.system.partition, a=[-0.65], b=[-0.85], c=[0], d=[0])
        
        itersToShow = 10
        
        cut_value = np.array([0.005, 0.005])
        
            
    # Compute all centers of regions associated with points
    x_init_centers = computeRegionCenters(np.array(x_init), Ab.system.partition, Ab.setup.floating_point_precision)

    # Filter to only keep unique centers
    x_init_unique = np.unique(x_init_centers, axis=0)
    
    state_idxs = [Ab.abstr['allCentersCubic'][tuple(c)] for c in x_init_unique 
                                   if tuple(list(c)) in Ab.abstr['allCentersCubic']]
    
    print(' -- Perform simulations for initial states:',state_idxs)
    
    mc_obj = MonteCarloSim(Ab, iterations = trajectoriesToCompute,
                               init_states = state_idxs )
    
    Ab.mc = {'reachability_probability': mc_obj.results['reachability_probability'],
                 'traces': mc_obj.traces }
    
    PRISM_reach = Ab.mdp.MAIN_DF['opt_reward'][state_idxs].to_numpy()
    empirical_reach = Ab.mc['reachability_probability']
    
    print('Probabilistic reachability (PRISM): ',PRISM_reach)
    print('Empirical reachability (Monte Carlo):',empirical_reach)
    
    performance_df = pd.DataFrame( {'PRISM reachability': PRISM_reach.flatten(),
                                    'Empirical reachability': empirical_reach.flatten() }, index=[case_id] )
    if writer != None:
        performance_df.to_excel(writer, sheet_name='Performance')
    
    traces = []
    
    for i in state_idxs:
        for j in range(itersToShow):
            traces += [Ab.mc['traces'][i][j]['x']]
            
    if Ab.setup.main['mode'] == 'Filter':
        belief_traces = {'mu': [], 'cov': []}
        for i in state_idxs:
            for j in range(itersToShow):
                belief_traces['mu'] += [Ab.mc['traces'][i][j]['bel_mu']]
                belief_traces['cov'] += [Ab.mc['traces'][i][j]['bel_cov']]
                
    else:
        belief_traces = None
    
    alltraces = []
    for i in state_idxs:
        for j in range(trajectoriesToCompute):
            alltraces += [Ab.mc['traces'][i][j]['x']]
    
    traces_df = pd.DataFrame(alltraces)
    traces_df.to_pickle(Ab.setup.directories['outputFcase']+'traces.pickle')
    
    if Ab.system.name == 'UAV_3D':
        if Ab.setup.main['iterative'] is False and Ab.setup.plotting['3D_UAV']:
        
            # Only plot trajectory plot in non-iterative mode (because it pauses the script)
            UAVplot3d_visvis( Ab.setup, Ab.system.spec, cut_value, traces ) 
    
    return performance_df, traces
    


def plot_trajectory_2D_animated(Ab, case_id, writer = None, itersToShow = 1):
    '''
    Ab : abstraction instance
        Full object of the abstraction being plotted for
    case_id : int
        Index for the current abstraction iteration
    writer : XlsxWriter
        Writer object to write results to Excel
    itersToShow : int
        Number of trajectories to plot
    
    '''

    from ..filterBasedAbstraction import MonteCarloSim

    x_init = Ab.system.x0
    
    cut_value = np.zeros(2)
    for i,d in enumerate(range(1, Ab.system.LTI['n'], 2)):
        if Ab.system.partition['nrPerDim'][d]/2 != round( Ab.system.partition['nrPerDim'][d]/2 ):
            cut_value[i] = 0
        else:
            cut_value[i] = Ab.system.partition['width'][d] / 2                

    # Compute all centers of regions associated with points
    x_init_centers = computeRegionCenters(np.array(x_init), Ab.system.partition, Ab.setup.floating_point_precision)
    
    # Filter to only keep unique centers
    x_init_unique = np.unique(x_init_centers, axis=0)
    
    state_idxs = [Ab.abstr['allCentersCubic'][tuple(c)] for c in x_init_unique 
                                   if tuple(list(c)) in Ab.abstr['allCentersCubic']]
    
    print(' -- Perform simulations for initial states:',state_idxs)
    
    mc_obj = MonteCarloSim(Ab, iterations = itersToShow,
                               init_states = state_idxs )
    
    traces = []
    
    for i in state_idxs:
        for j in range(itersToShow):
            traces += [mc_obj.traces[i][j]['x']]
            
    if Ab.setup.main['mode'] == 'Filter':
        belief_traces = {'mu': [], 'cov': []}
        for i in state_idxs:
            for j in range(itersToShow):
                belief_traces['mu'] += [mc_obj.traces[i][j]['bel_mu']]
                belief_traces['cov'] += [mc_obj.traces[i][j]['bel_cov']]

    #####

    min_delta = int(min(Ab.setup.all_deltas))

    i_show = tuple(Ab.setup.preset.plot_trajectory_2D)
    i_hide = tuple([i for i in range(Ab.system.LTI['n']) if i not in i_show])

    plot_times = np.arange(1, Ab.N+1)
    
    filenames = ['' for i in range(len(plot_times))]
        
    # Create list of error bounds
    error_bound_list = [dic['error_bound'] if 'error_bound' in dic else 0 for dic in Ab.km[Ab.system.base_delta].values()]
    
    for i,plot_time in enumerate(plot_times):
        filenames[i] = trajectoryPlot2D(i_show, i_hide, plot_time, Ab.N, Ab.setup, 
            Ab.model[min_delta], Ab.system.partition,
            Ab.system.spec, Ab.abstr, 
            error_bound_list, traces, belief_traces)

    img_array = []
    for filename in filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    video_filename = Ab.setup.directories['outputFcase']+'drone_trajectory_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_filename,fourcc, 1, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



def trajectoryPlot2D(i_show, i_hide, plot_time, N, setup, model, partition, spec, abstr, max_error_bound, 
              traces, belief_traces = None):
    '''
    Create 2D trajectory plots for the 2D UAV benchmark

    Parameters
    ----------
    i_show : list
        List of indices to show in plot
    i_hide : list
        List of indices to hide in plot
    plot_time : int
        Time step to plot results for
    N : int
        End time step
    setup : dict
        Setup dictionary.
    model : dict
        Main dictionary of the LTI system model.
    partition : dict
        Dictionary containing info about the partitioning
    spec : dict
        Dictionary containing info about specification
    abstr : dict
        Dictionay containing all information of the finite-state abstraction.
    max_error_bound : array
        Error bound by which regions are augmented

    Returns
    -------
    None.

    '''
    
    from scipy.interpolate import interp1d
    
    is1, is2 = i_show
    # ih1, ih2 = i_hide
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
    
    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(partition['width'])
    domainMax = width * np.array(partition['nrPerDim']) / 2
    
    min_xy = partition['origin'] - domainMax
    max_xy = partition['origin'] + domainMax
    
    major_ticks_x = np.arange(min_xy[is1]+1, max_xy[is1]+1, 4*width[is1])
    major_ticks_y = np.arange(min_xy[is2]+1, max_xy[is2]+1, 4*width[is2])
    
    minor_ticks_x = np.arange(min_xy[is1], max_xy[is1]+1, width[is1])
    minor_ticks_y = np.arange(min_xy[is2], max_xy[is2]+1, width[is2])
    
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    
    plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
    
    # Goal x-y limits
    ax.set_xlim(min_xy[is1], max_xy[is1])
    ax.set_ylim(min_xy[is2], max_xy[is2])
    
    # Draw goal regions
    for region in spec['goal'].values():
        
        lower  = region['limits'][:,0]
        size   = region['limits'][:,1] - region['limits'][:,0]
        
        lower_eps  = lower + max_error_bound[plot_time]
        size_eps   = size - 2*max_error_bound[plot_time]
        
        state = Rectangle(lower[[is1, is2]], width=size[is1], height=size[is2], 
                              color="green", alpha=0.3, linewidth=None)
        ax.add_patch(state)
        
        bound = Rectangle(lower_eps[[is1, is2]], width=size_eps[is1], height=size_eps[is2], 
                              edgecolor="green", linestyle='dashed', facecolor='None')
        ax.add_patch(bound)
        
        
    # Draw critical regions
    for region in spec['critical'].values():
        
        lower  = region['limits'][:,0]
        size   = region['limits'][:,1] - region['limits'][:,0]
        
        lower_eps  = lower - max_error_bound[plot_time]#, :]
        size_eps   = size + 2*max_error_bound[plot_time]#, :]
        
        state = Rectangle(lower[[is1, is2]], width=size[is1], height=size[is2], 
                              color="red", alpha=0.3, linewidth=None)
        ax.add_patch(state)
        
        bound = Rectangle(lower_eps[[is1, is2]], width=size_eps[is1], height=size_eps[is2], 
                              edgecolor="red", linestyle='dashed', facecolor='None')
        ax.add_patch(bound)
            
    # Add traces
    for i,trace in enumerate(traces):
        
        if len(trace) < 3:
            printWarning('Warning: trace '+str(i)+' has length of '+str(len(trace)))
            continue
        
        # Convert nested list to 2D array
        trace_array = np.array(trace)
        
        length = min(plot_time, np.shape(trace_array)[0])
        
        # Extract x,y coordinates of trace
        x = trace_array[:length, is1]
        y = trace_array[:length, is2]
        points = np.array([x,y]).T
        
        # Plot precise points
        plt.plot(*points.T, 'o', markersize=3, color="black");
        
        for j,point in enumerate(points):
            
            ax.text(point[0], point[1], str(j), c='red', verticalalignment='center', horizontalalignment='center')
        
            if belief_traces != None:
                mean = belief_traces['mu'][i][j][ [is1, is2] ]
                row_idx = col_idx = np.array([is1, is2])
                cov = belief_traces['cov'][i][j][ row_idx[:, None], col_idx ]
                
                confidence_ellipse(mean, cov, ax, n_std=1, edgecolor='gold',
                                   facecolor='gold', alpha=0.6)
        
        if length > 2:
        
            # Linear length along the line:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            
            # Interpolation for different methods:
            alpha = np.linspace(0, 1, 75)
            
            interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
            interpolated_points = interpolator(alpha)
            
            # Plot trace
            plt.plot(*interpolated_points.T, '-', color="blue", linewidth=0.5)
            
        elif length == 2:
            
            plt.plot(*points.T[:2, :], '-', color="blue", linewidth=0.5)
    
    # Set title
    ax.set_title("Trajectory up to time k="+str(plot_time), fontsize=8)
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'drone_trajectory_k='+str(plot_time)
    formats = setup.plotting['exportFormats']
    
    if 'png' not in formats:
        formats += ['png']
    
    for form in formats:
        if form != 'pdf' or plot_time == N:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight', dpi=300)
        
    # plt.show()
    
    return filename+'.png'
    
def UAVplot3d_visvis(setup, spec, cut_value, traces):
    '''
    Create 3D trajectory plots for the 3D UAV benchmark

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    spec : dict
        Dictionary containing info about specification
    cut_value : array
        Values to create the cross-section for
    traces : list
        Nested list containing the trajectories (traces) to plot for

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
    
    # Draw goal states
    for region in spec['goal'].values():
        
        center = np.mean( region['limits'], axis=1 )
        size   = (region['limits'][:,1] - region['limits'][:,0])[[0,2,4]]
        
        if center[1] == cut_value[0] and center[3] == cut_value[1] and center[5] == cut_value[2]:
        
            center_xyz = center[[0,2,4]]
            
            goal = vv.solidBox(tuple(center_xyz), scaling=tuple(size))
            goal.faceColor = (0,0.8,0,0.8)
            
    # Draw critical states
    for region in spec['critical'].values():
        
        center = np.mean( region['limits'], axis=1 )
        size   = (region['limits'][:,1] - region['limits'][:,0])[[0,2,4]]
        
        if center[1] == cut_value[0] and center[3] == cut_value[1] and center[5] == cut_value[2]:
        
            center_xyz = center[[0,2,4]]
            
            critical = vv.solidBox(tuple(center_xyz), scaling=tuple(size))
            critical.faceColor = (0.8,0,0,0.8)
    
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
        vv.plot(x,y,z, lw=0, mc=(102/255, 178/255, 255/255), ms='.', mw=12)
        
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
        vv.plot(xp,yp,zp, lw=5, lc=(102/255, 178/255, 255/255))
    
    ax.axis.xLabel = 'X'
    ax.axis.yLabel = 'Y'
    ax.axis.zLabel = 'Z'
    
    app = vv.use()
    f.relativeFontSize = 1.6
    # ax.position.Correct(dh=-5)
    vv.axis('tight', axes=ax)
    
    fig.position.w = 1000
    fig.position.h = 750
    
    im = vv.getframe(vv.gcf())
    
    ax.SetView({'zoom':0.042, 'elevation':25, 'azimuth':-35})
    
    filename = setup.directories['outputFcase'] + 'UAV_paths_screenshot.png'
    
    vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())
    
def load_traces_manual(Ab, paths, labels, idxs=[0]):
    '''
    Function to plot 3D UAV benhmark with multiple distinct cases

    Parameters
    ----------
    Ab : abstraction instance
        Full object of the abstraction being plotted for
    paths : list of string
        Paths to pickles file describing traces.
    labels : list of strings
        Labels for the instances to plot
    idxs : list of ints, optional
        Index of traces to be plotted. The default is 0.

    Returns
    -------
    None.

    '''
    
    if len(paths) != len(idxs):
        idxs = np.zeros(len(paths))
    
    traces_all = {}
    
    for i,path in enumerate(paths):
        traces_file = pd.read_pickle(path+'//traces.pickle')
        
        traces_all[i] = traces_file.loc[idxs[i]]
    
    cut_value = np.zeros(3)
    for i,d in enumerate(range(1, Ab.system.LTI['n'], 2)):
        if Ab.system.partition['nrPerDim'][d]/2 != round( Ab.system.partition['nrPerDim'][d]/2 ):
            cut_value[i] = 0
        else:
            cut_value[i] = Ab.system.partition['width'][d] / 2
        
    UAVplot3d_visvis_multi( Ab.setup, Ab.system.spec, cut_value, traces_all, labels) 
    
def UAVplot3d_visvis_multi(setup, spec, cut_value, traces_all, traces_labels):
    '''
    Create 3D trajectory plots for the 3D UAV benchmark

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    spec : dict
        Dictionary containing info about specification
    cut_value : array
        Values to create the cross-section for
    traces_a; : list
        Nested list containing the trajectories (traces) to plot for
   traces_all : list
        Nested list containing the trajectories (traces) to plot for
    traces_labels : list of string
        Legend labels for the instances to plot 

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
    
    # Draw goal states
    for region in spec['goal'].values():
        
        center = np.mean( region['limits'], axis=1 )
        size   = (region['limits'][:,1] - region['limits'][:,0])[[0,2,4]]
        
        if center[1] == cut_value[0] and center[3] == cut_value[1] and center[5] == cut_value[2]:
        
            center_xyz = center[[0,2,4]]
            
            goal = vv.solidBox(tuple(center_xyz), scaling=tuple(size))
            goal.faceColor = (0,0.8,0,0.8)
            
    # Draw critical states
    for region in spec['critical'].values():
        
        center = np.mean( region['limits'], axis=1 )
        size   = (region['limits'][:,1] - region['limits'][:,0])[[0,2,4]]
        
        if center[1] == cut_value[0] and center[3] == cut_value[1] and center[5] == cut_value[2]:
        
            center_xyz = center[[0,2,4]]
            
            critical = vv.solidBox(tuple(center_xyz), scaling=tuple(size))
            critical.faceColor = (0.8,0,0,0.8)
        
    trace_colors = [(102/255, 178/255, 255/255),
                    (204/255, 153/255, 255/255),
                    (255/255, 204/255, 229/255)]
    trace_styles = ['.', 'x', '*']
    trace_labels = tuple(np.repeat(traces_labels, 2))
    
    for j in range(max(len(traces) for traces in traces_all.values())):
    
        for i,traces in traces_all.items():
        
            if len(traces) <= j:
                continue
            else:
                trace = list(traces.loc[j].dropna())   
        
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
            vv.plot(x,y,z, lw=0, mc=trace_colors[i], ms=trace_styles[i], mw=12, label='test')
            
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
            vv.plot(xp,yp,zp, lw=5, lc=trace_colors[i])
        
    ax.axis.xLabel = 'X'
    ax.axis.yLabel = 'Y'
    ax.axis.zLabel = 'Z'
    
    app = vv.use()
    
    f.relativeFontSize = 1.6
    vv.axis('tight', axes=ax)
    
    fig.position.w = 1000
    fig.position.h = 750
    
    im = vv.getframe(vv.gcf())
    
    ax.SetView({'zoom':0.03, 'elevation':70, 'azimuth':20})
    
    ax.legend = trace_labels
    
    if 'outputFcase' in setup.directories:
    
        filename = setup.directories['outputFcase']+'UAV_multi_instances.png'
        
    else:
        
        filename = setup.directories['outputF'] + 'UAV_multi_instances.png'
    
    vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())
    
def plot_heatmap(Ab, plot_values, filename, vrange=[-0.2,0.2], cmap=sns.diverging_palette(220, 20, sep=1, as_cmap=True)):
    '''
    Create heat map for the BAS benchmarks

    Parameters
    ----------
    Ab : abstraction instance
        Full object of the abstraction being plotted for

    Returns
    -------
    None.

    '''

    # Create dataframes of values to plot
    x_nr = Ab.system.partition['nrPerDim'][Ab.setup.preset.plot_heatmap[0]]
    y_nr = Ab.system.partition['nrPerDim'][Ab.setup.preset.plot_heatmap[1]]

    v = [1 for i in range(Ab.system.LTI['n'])]
    v[Ab.setup.preset.plot_heatmap[0]] = x_nr
    v[Ab.setup.preset.plot_heatmap[1]] = y_nr

    cut_centers = definePartitions(Ab.system.LTI['n'], v, 
        Ab.system.partition['width'], 
        Ab.system.partition['origin'], onlyCenter=True)

    cut_values = np.zeros((x_nr, y_nr))
    cut_coords = np.zeros((x_nr, y_nr, Ab.system.LTI['n']))
    
    cut_idxs = [Ab.abstr['allCentersCubic'][tuple(c)] for c in cut_centers]
                                #if tuple(c) in Ab.abstr['allCentersCubic']]

    for i,(idx,center) in enumerate(zip(cut_idxs, cut_centers)):
        
        j = i % y_nr
        k = i // y_nr
        
        cut_values[k,j] = plot_values[idx]
        cut_coords[k,j,:] = center

    if Ab.system.name == 'UAV_2D':
        cut_df = pd.DataFrame( cut_values, index=cut_coords[:,0,0], columns=cut_coords[0,:,2] )
    else:
        cut_df = pd.DataFrame( cut_values, index=cut_coords[:,0,0], columns=cut_coords[0,:,1] )

    fig = plt.figure(figsize=cm2inch(9, 8))
    
    # If regions are non-skewed, use standard Seaborn heatmap.
    if not Ab.setup.main['skewed']:
        ax = sns.heatmap(cut_df.T, cmap=cmap, #YlGnBu
                 vmin=vrange[0], vmax=vrange[1], fmt='.1f')
        
        ax.invert_yaxis()
        ax.figure.axes[-1].yaxis.label.set_size(20)
    
    else:
        printWarning('Heatmap for system with skewed regions not (yet) supported.')
        return
    
    # ax.set_xticklabels(xticks, size = 13)
    # ax.set_yticklabels(yticks, size = 13)
    ax.set_xlabel('x_1', fontsize=15)
    ax.set_ylabel('x_2', fontsize=15)
    if Ab.setup.main['mode'] == 'Filter':
        ax.set_title("Filter-based abstraction",fontsize=18)
    else:
        ax.set_title("N = "+str(Ab.setup.scenarios['samples']),fontsize=18)
    
    # Set tight layout
    fig.tight_layout()

    # Save figure
    for form in Ab.setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    # plt.show()
    
def control_plot(Ab, k=0):
    
    policy = Ab.mdp.MAIN_DF['opt_action']
    
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
        
    plt.xlabel('$x_1$', labelpad=0)
    plt.ylabel('$x_2$', labelpad=0)
    
    for i, region in Ab.abstr['P'].items():
        
        state = region['center']
        
        action = policy[i][k]
        
        if action != -1:
            state_hat = Ab.abstr['target']['d'][action]
            
            diff = state_hat - state
            
            plt.arrow(state[0], state[1], diff[0], diff[1], width=0.1, 
                      linewidth=0, head_width = 1, head_length = 1)
            
    # Set title
    ax.set_title("Control under filter-based abstraction", fontsize=8)
    
    # Set axis limits
    ax.set_xlim(-22, 22)
    ax.set_xlim(-22, 22)
    
    # Set tight layout
    fig.tight_layout()
            
    # plt.show()
    