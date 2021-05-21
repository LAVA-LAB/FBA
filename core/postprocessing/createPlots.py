#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mpl
# mpl.use("pgf")
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.spatial import ConvexHull
import scipy.interpolate
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors

from ..commons import printWarning, mat_to_vec, cm2inch


def createPartitionPlot(j, delta_plot, setup, model, \
                        abstr, allOriginPointsNested, predecessor_set):
    '''
    Create partition plot for the current filter-based abstraction instance.

    Parameters
    ----------
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
    allOriginPointsNested : list
        Nested lists containing all origin points of all regions.
    mu_inv_hulls : list
        List containing all inverse reachability hulls.
    enabledPolypoints : list
        List of which states are contained in the inv. reachability hull.

    Returns
    -------
    None.

    '''
    
    # If number of dimensions is 2, create 2D plot
    if model.n <= 2:
        
        #### Partition plot v1
        
        fig = plt.figure(figsize=cm2inch(12, 7))
        ax = fig.add_subplot(111)
        
        plt.xlabel('$x_1$', labelpad=0)
        plt.ylabel('$x_2$', labelpad=-10)
        
        for k,poly in enumerate(allOriginPointsNested):

            # Convert poly list to numpy array            
            polyMat = np.array(poly)
            
            # Plot partitions and label
            ax.text(abstr['P'][k]['center'][0], abstr['P'][k]['center'][1], k, \
                      verticalalignment='center', horizontalalignment='center' )  
            hull = ConvexHull(polyMat, qhull_options='QJ')
            ax.plot(polyMat[hull.vertices,0], polyMat[hull.vertices,1], lw=1)
            ax.plot([polyMat[hull.vertices[0],0], polyMat[hull.vertices[-1],0]], \
                      [polyMat[hull.vertices[0],1], polyMat[hull.vertices[-1],1]], lw=1)
            
        for k,target_point in enumerate(abstr['target']['d']):
                
            # Plot target point
            plt.scatter(target_point[0], target_point[1], c='k', s=6)
        
        if setup.plotting['partitionPlot_plotHull']:
            # Plot convex hull of the preimage of the target point
            hull = ConvexHull(predecessor_set, qhull_options='QJ')
            ax.plot(predecessor_set[hull.vertices,0], predecessor_set[hull.vertices,1], 'ro--', lw=1) #2
            ax.plot([predecessor_set[hull.vertices[0],0], predecessor_set[hull.vertices[-1],0]], \
                      [predecessor_set[hull.vertices[0],1], predecessor_set[hull.vertices[-1],1]], 'ro--', lw=1) #2
            
        # Set tight layout
        fig.tight_layout()
                    
        # Save figure
        filename = setup.directories['outputF']+'partitioning_'+str(delta_plot)
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
        
        ###########################
        #### Partition plot v2 ####
        fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
        
        plt.xlabel('$x_1$', labelpad=0)
        plt.ylabel('$x_2$', labelpad=-10)

        domainMax = np.array(model.setup['partition']['nrPerDim']) * np.array(model.setup['partition']['nrPerDim']) / 2
        
        min_xy = model.setup['partition']['origin'] - domainMax
        max_xy = model.setup['partition']['origin'] + domainMax
        
        width = model.setup['partition']['width']
        
        major_ticks_x = np.arange(min_xy[0]+1, max_xy[0]+1, 4*width[0])
        major_ticks_y = np.arange(min_xy[1]+1, max_xy[1]+1, 4*width[1])
        
        minor_ticks_x = np.arange(min_xy[0], max_xy[0]+1, width[0])
        minor_ticks_y = np.arange(min_xy[1], max_xy[1]+1, width[1])
        
        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(minor_ticks_y, minor=True)
        
        for axi in (ax.xaxis, ax.yaxis):
            for tic in axi.get_minor_ticks():
                tic.tick1On = tic.tick2On = False
        
        # plt.grid(which='major', color='#CCCCCC', linewidth=0.3)
        plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
        
        # Goal x-y limits
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        
        # Draw goal states
        for goal in abstr['goal']:
            goal_lower = abstr['P'][goal]['low']
            goalState = Rectangle(goal_lower[0:2], width=width[0], height=width[1], color="green", alpha=0.3, linewidth=None)
            ax.add_patch(goalState)
        
        # Draw critical states
        for critState in abstr['critical']:
            critStateLow = abstr['P'][critState]['low']
            criticalState = Rectangle(critStateLow[0:2], width=width[0], height=width[1], color="red", alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)
        
        with plt.rc_context({"font.size": 5}):        
            # Draw every X-th label
            skip = 15
            for i in range(0, abstr['nr_regions'], skip):
                ax.text(abstr['P'][i]['center'][0], abstr['P'][i]['center'][1], i, \
                          verticalalignment='center', horizontalalignment='center' ) 
                
        # Set tight layout
        fig.tight_layout()
        
        # Save figure
        filename = setup.directories['outputF']+'partitioning_v2'
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
    
    # If number of dimensions is 3, create 3D plot
    elif model.n == 3:
        
        fig = plt.figure(figsize=cm2inch(5.33, 4))
        ax = fig.add_subplot(111, projection="3d")
        
        for k,poly in enumerate(allOriginPointsNested):
    
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
    """Set 3D plot axes to equal scale.

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
    Create the result plots for the filter-based abstraction instance.

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
            
            ax.set_box_aspect([1,1,1])
            set_axes_equal(ax)
            ax_comb[fig_ind].set_box_aspect([1,1,1])
            set_axes_equal(ax_comb[fig_ind])
        
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
    
def createTimegroupingPlot(setup, plot, model, results, abstr, max_delta):
    '''
    Create plot that shows both the actions and the associated delta values.

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    plot : dict
        Dictionary containing info about plot settings
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
    
    ### NOTE: THIS FUNCTION IS OUTDATED (DOESNT WORK ANYMORE.)
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    
    # If number of dimensions is 2, create 2D plot
    if model.n <= 2:
        
        #### Retreive results
        
        # Retreive grid size
        m = setup['PartitionsFromCenter']*2+1
        
        #get discrete colormap
        data_all = results['optimal_delta']
        cmap = plt.get_cmap('Greys', max_delta+4-np.min(data_all)+1)
        
        # Retreive highest delta value of any action used
        highest_delta = np.max(data_all)
        
        
        #### Single plot for time k=0
        
        fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
        
        plt.xlabel('$x_1$', labelpad=0)
        plt.ylabel('$x_2$', labelpad=-10)
    
        k = 2
    
        # Shortcut to data (flipped to make values correspond to axes)
        data = results['optimal_delta'][k,:]                      
        data_reshape = np.flip(np.reshape(data, (m,m)), axis=0)
        
        # set limits .5 outside true range
        mat = ax.imshow(data_reshape,cmap=cmap,vmin = np.min(data_all)-.5, vmax = max_delta+4+.5)
        
        # Retreive sizes of regions
        origin_x = setup['stateSpaceOrigin'][0]
        origin_y = setup['stateSpaceOrigin'][1]
        min_x = origin_x - setup['stateSpaceDomainMax']
        max_x = origin_x + setup['stateSpaceDomainMax']
        min_y = origin_y - setup['stateSpaceDomainMax']
        max_y = origin_y + setup['stateSpaceDomainMax']
        
        width = setup['stateSpaceDomainMax'] / (setup['PartitionsFromCenter']+0.5)
        
        major_ticks_x = np.arange(min_x+1, max_x+1, int(4*width))
        major_ticks_y = np.arange(min_y+1, max_y+1, int(4*width))[::-1]
        
        # Set ticks
        plt.xticks(np.arange(0,m,4), major_ticks_x)
        plt.yticks(np.arange(0,m,4), major_ticks_y)
        
        # Plot individual actions as arrows
        for i in range(highest_delta,0,-1):
            # Create boolean list for actions associated with current delta
            actionBool = [j == i for j in data]
            
            # Retreive relative actions corresponding to this delta
            action_data = results['optimal_policy'][k,actionBool]
            origin_state = np.arange(0,len(abstr['P']))[actionBool]
            
            arrow_frequency = 5
            
            # Plot these actions as arrows
            for l,act in enumerate(action_data):
                
                if origin_state[l] % arrow_frequency != 0:
                    continue
                
                # Calculate state ID for arrow to draw
                x = origin_state[l] % m
                y = m-1 - origin_state[l] // m
                dx = act % m - x
                dy = m-1 - act // m - y
                
                alpha = 1 #max(0.2, min(0.5, (0.7*results['optimal_reward'][0,origin_state[l]])))
                edgecol = colors['maroon'] #to_rgba('red', 0.5) #(0.7,0,0)
                facecol = colors['forestgreen'] #to_rgba('cyan', 1) #'c'
                
                # Draw arrow
                ax.arrow(x,y,dx,dy, lw=0.05, head_width=0.5, head_length=0.7, 
                    length_includes_head=True, edgecolor=edgecol, facecolor=facecol, alpha=alpha)
        
        # Set tight layout
        fig.tight_layout()
        
        fig.subplots_adjust(right=1.15)
        cbar_ax = fig.add_axes([0.93, 0.25, 0.05, 0.6])
        
        #tell the colorbar to tick at integers
        fig.colorbar(mat, cax=cbar_ax, ticks=np.arange(np.min(data_all),np.max(data_all)+1))
        
        # Save figure
        filename = setup.directories['outputFcase']+'policy_action_delta_time0'
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
        #### Combined plot for multiple points in time
        
        fig_comb    = plt.figure(figsize=cm2inch(16,5.33))
        fig_ind     = 0
        ax_comb     = dict()
        
        time_list = [ [plot['N'][key], plot['T'][key]] for key in plot['N'].keys() ]
        
        # For each of the time slots, create a subplot
        for [k,t] in time_list:
            
            # Increase combo figure index
            fig_ind += 1
                
            # Create combo subplot
            ax_comb[fig_ind]  = fig_comb.add_subplot(1,4,fig_ind)
        
            # Shortcut to data (flipped to make values correspond to axes)
            data = results['optimal_delta'][k,:]                      
            data_reshape = np.flip(np.reshape(data, (m,m)), axis=0)
            
            # set limits .5 outside true range
            mat = ax_comb[fig_ind].imshow(data_reshape,cmap=cmap,vmin = np.min(data_all)-.5, vmax = np.max(data_all)+.5)
            
            # Format subplot
            ax_comb[fig_ind].set_title('Time k='+str(t))
            ax_comb[fig_ind].set_xlabel('$x_1$', labelpad=-2)
            ax_comb[fig_ind].set_ylabel('$x_2$', labelpad=-4)
            
            xlabels = np.arange(-20,21,8)
            ylabels = np.arange(20,-21,-8)
            
            plt.xticks(np.arange(0,m,4), xlabels)
            plt.yticks(np.arange(0,m,4), ylabels)
            
            # Plot individual actions as arrows
            for i in range(highest_delta,0,-1):
                # Create boolean list for actions associated with current delta
                actionBool = [j == i for j in data]
                
                # Retreive relative actions corresponding to this delta
                action_data = results['optimal_policy'][k,actionBool]
                origin_state = np.arange(0,len(abstr['P']))[actionBool]
                
                # Plot these actions as arrows
                for l,act in enumerate(action_data):
                    # Calculate state ID for arrow to draw
                    x = origin_state[l] % m
                    y = m-1 - origin_state[l] // m
                    dx = act % m - x
                    dy = m-1 - act // m - y
                    
                    # Draw arrow
                    ax_comb[fig_ind].arrow(x,y,dx,dy, lw=0.05, head_width=0.3, head_length=0.2, \
                        length_includes_head=True, color='r', alpha=(0.1+0.3*results['optimal_reward'][k,origin_state[l]]))
    
        # Set tight layout
        fig_comb.tight_layout()
        
        fig_comb.subplots_adjust(right=1.15)
        cbar_ax = fig_comb.add_axes([0.9, 0.2, 0.05, 0.6])
        
        #tell the colorbar to tick at integers
        fig_comb.colorbar(mat, cax=cbar_ax, ticks=np.arange(np.min(data_all),np.max(data_all)+1))
                    
        # Save figure
        filename = setup.directories['outputFcase']+'policy_action_delta_over_time'
        for form in setup.plotting['exportFormats']:
            plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    else:
        print('Number of dimensions is larger than 2, so partition plots omitted')
        
def UAVplot2D(setup, model, abstr, traces, cut_value):
    
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
            tic.tick1On = tic.tick2On = False
    
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
            printWarning('Warning: trace',i,'has length of',len(trace))
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
    
      
def UAVplot3D(setup, model, abstr, traces, cut_value):
    
    import numpy as np
    
    from scipy.interpolate import interp1d
    
    ix = 0
    iy = 2
    iz = 4
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
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
            
            plotCube(ax, center_xyz, regionWidth_xyz, 0.1, 'g')
            
    # Draw critical states
    for crit in abstr['critical']:
        
        critState = abstr['P'][crit]
        if critState['center'][1] == cut_value[0] and critState['center'][3] == cut_value[1] and critState['center'][5] == cut_value[2]:
        
            center_xyz = np.array([critState['center'][0], 
                                   critState['center'][2], 
                                   critState['center'][4]])    
        
            plotCube(ax, center_xyz, regionWidth_xyz, 0.2, 'r')   
        
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
        
        print(points)
        
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
        
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
           
    width = np.array(model.setup['partition']['width'])
    domainMax = width * np.array(model.setup['partition']['nrPerDim']) / 2
    min_xyz = model.setup['partition']['origin'] - domainMax
    max_xyz = model.setup['partition']['origin'] + domainMax
    
    ax.set_xlim(min_xyz[ix], max_xyz[ix])
    ax.set_ylim(min_xyz[iy], max_xyz[iy])
    ax.set_zlim(min_xyz[iz], max_xyz[iz])
    
    # Set tight layout
    fig.tight_layout()
    
    # Save figure
    filename = setup.directories['outputFcase']+'drone_trajectory'
    for form in setup.plotting['exportFormats']:
        plt.savefig(filename+'.'+str(form), format=form, bbox_inches='tight')
        
    plt.show()
    
def UAVplot3d_visvis(setup, model, abstr, traces, cut_value):
    
    from scipy.interpolate import interp1d
    import visvis as vv
    
    app = vv.use()
    
    f = vv.clf()
    a = vv.cla()
    
    ix = 0
    iy = 2
    iz = 4
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
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
        
        print(points)
        
        # Plot precise points
        #plt.plot(x, y, z, lw=4, color="black");
        
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
        vv.plot(xp,yp,zp, lw=1);
        # plt.plot(x_values, y_values, color="blue")
    
    # angle = np.linspace(0, 6*np.pi, 1000)
    # x = np.sin(angle)
    # y = np.cos(angle)
    # z = angle / 6.0
    # vv.plot(x, y, z, lw=10)
    
    # angle += np.pi*2/3.0
    # x = np.sin(angle)
    # y = np.cos(angle)
    # z = angle / 6.0 - 0.5
    # vv.plot(x, y, z, lc ="r", lw=10)
    
    app.Run()
    
    
def plotCube(ax, center, width, alpha, color):
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    alpha = float(alpha)
    color = str(color)
    
    points = np.array([ [-1, -1, -1],
                        [1, -1, -1 ],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1 ],
                        [1, 1, 1],
                        [-1, 1, 1] ])
    
    Z = center + points * 0.5 * width

    r = [-1,1]
    
    X, Y = np.meshgrid(r, r)
    
    # plot vertices
    # ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
    
    # list of sides' polygons of figure
    verts = [[Z[0],Z[1],Z[2],Z[3]],
     [Z[4],Z[5],Z[6],Z[7]], 
     [Z[0],Z[1],Z[5],Z[4]], 
     [Z[2],Z[3],Z[7],Z[6]], 
     [Z[1],Z[2],Z[6],Z[5]],
     [Z[4],Z[7],Z[3],Z[0]]]
    
    cube = Poly3DCollection(verts, 
     facecolors=color, linewidths=1, edgecolors=color, alpha=alpha)
    
    # plot sides
    ax.add_collection3d(cube)