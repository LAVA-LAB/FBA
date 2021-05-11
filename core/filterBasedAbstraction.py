#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import copy
import csv

from scipy.spatial import Delaunay
from scipy.stats import mvn

from .mainFunctions import createLTImodel, createMDP, kalmanFilter, \
    computeGaussianProbabilities, definePartitions, defineDistances, \
    in_hull, makeModelFullyActuated
from .commons import tic, toc, ticDiff, tocDiff, \
    nearest_point_in_hull, table, floor_decimal
from .postprocessing.createPlots import createPartitionPlot
from core.postprocessing.plotTrajectory import plotTrajectory

'''
------------------------------------------------------------------------------
Main filter-based abstraction object definition
------------------------------------------------------------------------------
'''

class filterBasedAbstraction(object):
    '''
    Main filter-based abstraction object.
    '''

    def __init__(self, setup):
        '''
        Initialize the filter-based abstraction object.

        Parameters
        ----------
        setup : dict
        Setup dictionary.

        Returns
        -------
        None.

        '''

        # Copy setup to internal variable
        self.setup = setup
        self.setup['partitionWidth'] = self.setup['stateSpaceDomainMax'] / (self.setup['PartitionsFromCenter']+0.5)
        
        # Define empty dictionary for monte carlo simulations (even if not used)
        self.mc = dict()   
        
        # Start timer
        tic()
        ticDiff()
        self.time = dict()
        
        self.basemodel = dict()
        self.basemodel['tau']= self.setup['samplingTime']    # Sampling time
        
        self.basemodel = createLTImodel(self.basemodel, self.setup, typ=self.setup['modelType'])
        
        # Random disturbance parameters
        self.basemodel['noise']          = dict()
        self.basemodel['noise']['w_cov'] = np.eye(np.size(self.basemodel['A'],1))*self.setup['sigma_w_value']
        self.basemodel['noise']['v_cov'] = np.eye(np.size(self.basemodel['C'],0))*self.setup['sigma_v_value']
        self.basemodel['noise']['w_mean']    = self.basemodel['W'].flatten()
        
        # Retreive system dimensions
        self.basemodel['n']     = np.size(self.basemodel['A'],1)   # Nr of states
        self.basemodel['r']     = np.size(self.basemodel['C'],0)   # Nr of measurements (outputs)
        
        # Compute the observability matrix
        self.basemodel['observability_matrix'] = np.vstack([self.basemodel['C'] @ \
                np.linalg.matrix_power(self.basemodel['A'], i) for i in range(0,self.basemodel['n'])])
        self.basemodel['observability_matrix_rank'] = np.linalg.matrix_rank(self.basemodel['observability_matrix'])
            
        # If the observability matrix is full rank, system is  fully observable
        if self.basemodel['observability_matrix_rank'] == self.basemodel['n']:
            self.basemodel['partially_observable'] = False
            
            print(' -- System is fully observable')
        else:
            self.basemodel['partially_observable'] = True
            
            print(' -- System is partially observable; rank is', self.basemodel['observability_matrix_rank'])
            
        self.model = dict()
        for delta in self.setup['deltas']:
            if delta == 1:
                self.model[delta] = self.basemodel
            else:
                self.model[delta] = makeModelFullyActuated(copy.deepcopy(self.basemodel), manualDimension = delta)
                
            # Determine inverse A matrix
            self.model[delta]['A_inv']  = np.linalg.inv(self.model[delta]['A'])
            
            # Determine pseudo-inverse B matrix
            self.model[delta]['B_pinv'] = np.linalg.pinv(self.model[delta]['B'])
            
            # Retreive system dimensions
            self.model[delta]['p']      = np.size(self.model[delta]['B'],1)   # Nr of inputs
        
            # Control limitations
            self.model[delta]['uMin']   = np.array( self.setup['uMin'] * self.model[delta]['p'] )
            self.model[delta]['uMax']   = np.array( self.setup['uMax'] * self.model[delta]['p'] )
        
        # Simulation end time is the end time
        self.T  = self.setup['endTime']
        
        # Number of simulation time steps
        self.N = int(self.T/self.basemodel['tau'])
        
        self.time['0_init'] = tocDiff(False)
        print('Object initialized - time:',self.time['0_init'])
        
        ''' 
        -----------------------------------------------------------------------
        DEFINE THE DISCRETE STATE PARTITION
        -----------------------------------------------------------------------
        '''
        
        # Define partitioning of state-space
        self.abstr                  = dict()
        self.abstr['P'], self.abstr['center'] = definePartitions(self.basemodel['n'], self.setup['PartitionsFromCenter'], 
                                                                 self.setup['partitionWidth'], self.setup['stateSpaceOrigin'])
        self.abstr['nr_regions'] = len(self.abstr['P'])
        
        if self.setup['modelType'] == 'thermal2zone':
            goal = self.setup['goal_state_center']
            self.abstr['goal'] = [ i for i,region in self.abstr['P'].items()
                                       if region['center'][0:(self.basemodel['n']-1)] == goal ]
        else:
            self.abstr['goal'] = [self.abstr['center']]
            critical = self.setup['critical_state_center']
            self.abstr['critical'] = [ i for i,region in self.abstr['P'].items()
                                       if region['center'][0]-region['center'][1] > critical ]
        
        print(' -- Number of regions:',self.abstr['nr_regions'])
        print(' -- Goal regions are:',self.abstr['goal'])
        print(' -- Critical regions are:',self.abstr['critical'])
        
        for i in range(self.abstr['nr_regions']):
            
            # Determine difference in coordinates
            self.abstr['P'][i]['distances'] = defineDistances(self.abstr['P'], i)
            
            # Determine Euclidean distances
            self.abstr['P'][i]['absDistances'] = np.linalg.norm(self.abstr['P'][i]['distances'], axis=1)
            self.abstr['P'][i]['absDistSort'] = np.argsort( self.abstr['P'][i]['absDistances'] )

        self.time['1_partition'] = tocDiff(False)
        print('Discretized states defined - time:',self.time['1_partition'])

        ''' 
        -----------------------------------------------------------------------
        DETERMINE WHICH ACTIONS ARE ACTUALLY ENABLED
        -----------------------------------------------------------------------
        '''
        
        internal = dict()
                
        # Column widths for tabular prints
        if self.setup['verbose']:
            col_width = [8,8,8,46]
            tab = table(col_width)
            
        # Create all combinations of n bits, to reflect combinations of all lower/upper bounds of partitions
        internal['bitCombinations'] = list(itertools.product([0, 1], repeat=self.basemodel['n']))
        internal['bitRelation']     = ['low','upp']
        
        # Calculate all corner points of every partition.
        # Every partition has an upper and lower bounnd in every state (dimension).
        # Hence, the every partition has 2^n corners, with n the number of states.
        internal['allOriginPointsNested'] = [[[
                                    self.abstr['P'][i][internal['bitRelation'][bit]][bitIndex] for bitIndex,bit in enumerate(bitList)
                                ] for bitList in internal['bitCombinations'] 
                                ] for i in self.abstr['P']
                                ]
        self.abstr['allOriginPoints'] = np.concatenate( internal['allOriginPointsNested'] )
        
        # Create hull of the full state space
        self.abstr['fullStateSpaceHull'] = Delaunay(self.abstr['allOriginPoints'], qhull_options='QJ')
        
        # Set default target points to the center of every region
        self.abstr['targetPoints'] = [self.abstr['P'][j]['center'] for j in range(self.abstr['nr_regions'])]
        targetPointTuples = [tuple(point) for point in self.abstr['targetPoints']]        
        self.abstr['targetPointsInv'] = dict(zip(targetPointTuples, range(self.abstr['nr_regions'])))
        
        # If there is only one target region, we can use a dynamic target point
        if np.size(self.abstr['goal']) == 1:
        
            # Define the continuous goal state
            goal = self.abstr['P'][ self.abstr['goal'][0] ]['center'][0]
            
            # Define nearest target points to the goal state
            if setup['dynamicTargetPoint']:
                self.abstr['targetPoints'] = [nearest_point_in_hull(internal['allOriginPointsNested'][j], goal) for
                                                   j in range(self.abstr['nr_regions'])]
        
        self.abstr['enabledActions']     = dict()
        self.abstr['enabledActions_inv'] = dict()
        self.abstr['waitEnabled']        = dict()
        
        internal['enabled_polypoints']   = dict()
        internal['mu_inv_hull']          = dict()
        
        # For every time step grouping value in the list
        for delta_idx,delta in enumerate(self.setup['deltas']):
            
            ### Define control hull
            # Determine the set of extremal control inputs
            u = [[self.model[delta]['uMin'][i], self.model[delta]['uMax'][i]] for i in range(self.model[delta]['p'])]
            
            # Determine the inverse image of the extreme control inputs from target point
            x_inv_area = np.zeros((2**self.model[delta]['p'], self.model[delta]['n']))
            for i,elem in enumerate(itertools.product(*u)):
                list_elem = list(elem)
                
                # Calculate inverse image of the current extreme control input
                x_inv_area[i,:] = - self.model[delta]['A_inv'] @ \
                    (self.model[delta]['B'] @ np.array(list_elem).T + self.model[delta]['noise']['w_mean'])   
                    
            x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
            
            print(' -- Inverse hull created for delta',delta)
            
            # Print header row
            if self.setup['verbose']:
                tab.print_row(['ACTION','DELTA','STATUS'], head=True)
            
            # Empty data structure to store the enabled actions
            self.abstr['enabledActions'][delta] = [[] for 
                               i in range(self.abstr['nr_regions'])]
            
            self.abstr['enabledActions_inv'][delta] = [[] for 
                               i in range(self.abstr['nr_regions'])]
            
            self.abstr['waitEnabled'][delta] = np.full(self.abstr['nr_regions'],
                                                       False, dtype=bool)
            
            # Define dictionaries to sture points in the preimage of a state, and the
            # corresponding polytope points
            
            internal['total_actions'] = 0
            
            # For every (target) partition
            for partition_id in range(self.abstr['nr_regions']):
                
                # Check if this region is a critical state
                if partition_id in self.abstr['critical']:
                    # If so, continue with next region
                    continue
                
                targetPoint = self.abstr['targetPoints'][partition_id]
                    
                # Shift the origin points (instead of the target point)
                originPointShift = self.model[delta]['A_inv'] @ np.array(targetPoint).T
                originPointsShifted = [row - originPointShift for row in self.abstr['allOriginPoints']]
                
                rows = np.shape(x_inv_area)[0]
                
                # Shift the inverse hull to account for the specific target point
                internal['mu_inv_hull'][(partition_id,delta)] = x_inv_area + np.array([list(originPointShift)]*rows)
                internal['polypoints_vec'] = in_hull(originPointsShifted, x_inv_hull)
          
                # Map the enabled corner points of the partitions to actual partitions
                internal['enabled_polypoints'][(partition_id,delta)] = np.reshape(  internal['polypoints_vec'], (self.abstr['nr_regions'], 2**self.basemodel['n']))
                
                # Polypoints contains the True/False results for all corner points of every partition
                # An action is enabled in a state if it is enabled in all its corner points
                internal['enabled_actions']    = np.all(internal['enabled_polypoints'][(partition_id,delta)] == True, axis = 1)
                
                # Retreive the ID's of all enabled actions (for printing purposes)
                internal['enabled_actions_ID'] = np.where(internal['enabled_actions'] == True)[0]
                
                # Remove critical states from the list of enabled actions
                internal['enabled_actions_ID'] = np.setdiff1d(internal['enabled_actions_ID'], self.abstr['critical'])
                
                if self.setup['verbose']:
                    if partition_id in self.abstr['goal']:
                        add = ' (G)'
                    else:
                        add = ''
                    tab.print_row([str(partition_id)+add, delta, 
                                   'Enabled in '+str(len(internal['enabled_actions_ID']))+' states; sum is '+\
                                   str(sum(internal['polypoints_vec']))])
                
                if len(internal['enabled_actions_ID']) > 0:
                    internal['total_actions'] += 1
                
                self.abstr['enabledActions_inv'][delta][partition_id] = internal['enabled_actions_ID']
                
                for origin in internal['enabled_actions_ID']:
                    self.abstr['enabledActions'][delta][origin] += [partition_id]
                    
                    if origin == partition_id:
                        self.abstr['waitEnabled'][delta][origin] = True
                        
            print(' -- ',internal['total_actions'],'actions enabled for',delta)
                
        # Store the dictionary of origin points for each partition
        for i in self.abstr['P']:
            self.abstr['P'][i]['polypoints'] = internal['allOriginPointsNested'][i]
            self.abstr['P'][i]['hull'] = Delaunay(self.abstr['P'][i]['polypoints'], qhull_options='QJ')
        
        # Generate partition plot for the goal state, also showing the pre-image
        if self.setup['partitionPlot']:
            
            for delta_plot in self.setup['deltas']:
                createPartitionPlot(self.abstr['center'], delta_plot, self.setup, \
                            self.model[delta_plot], self.abstr, internal['allOriginPointsNested'], \
                            internal['mu_inv_hull'], internal['enabled_polypoints'])
        
        self.time['2_enabledActions'] = tocDiff(False)
        print('Enabled actions define - time:',self.time['2_enabledActions'])
        
    def KalmanPrediction(self):
        '''
        -----------------------------------------------------------------------
        KALMAN FILTER IMPLEMENTATION
        -----------------------------------------------------------------------
        '''
        
        # Column widths for tabular prints
        if self.setup['verbose']:
            col_width = [8,8,46]
            tab = table(col_width)
        
        self.km = dict()
        
        if self.setup['LIC']:
            # Determine maximum threshold for measure on covariance over time
            maxA = self.setup['LICMaxA']
            minA = self.setup['LICMinA']
            self.km['maxEllipseSize'] = [i*minA + (1-i)*maxA for i in np.arange(0,1,1/self.N)]
            
        for delta in self.setup['deltas']:
            
            # Print header row
            if self.setup['verbose']:
                tab.print_row(['DELTA','K','STATUS'], head=True)
            
            # Kalman filter list definitions
            self.km[delta]                      = dict()
            self.km[delta]['waiting_time']      = dict()
            self.km[delta]['cov_pred']          = [None] * (self.N+delta)
            self.km[delta]['cov']               = [None] * (self.N+delta)
            self.km[delta]['cov_tilde']         = [None] * (self.N+delta)
            self.km[delta]['cov_tilde_measure'] = [None] * (self.N+delta)
            self.km[delta]['K_gain']            = [None] * (self.N+delta)
            
            # Initial belief of Kalman filter
            for i in range(delta):
                self.km[delta]['cov'][i] = np.eye(self.basemodel['n'])*self.setup['kf_cov0']
            
            # For every time step in the finite horizon
            for k in range(self.N):
                
                # Set waiting time to zeor by default
                self.km[delta]['waiting_time'][k]  = 0
                
                # Perform Kalman filter prediction at current time k for delta-type action
                self.km[delta]['cov_pred'][k+delta], self.km[delta]['K_gain'][k+delta], \
                self.km[delta]['cov_tilde'][k+delta], self.km[delta]['cov'][k+delta], \
                self.km[delta]['cov_tilde_measure'][k+delta] = \
                    kalmanFilter(self.model[delta], self.km[delta]['cov'][k])
                
                # If local information controller is active
                if self.setup['LIC']:
                    # Then, a "worst-case" covariance matrix is used, and we
                    # force the actual covariance always to below this
                    
                    # Set the worst-case covariance matrix to the circle with  
                    # radius equal to the maximum measure (i.e. threshold)
                    self.km[delta]['cov_tilde'][k+delta] = \
                        np.eye(self.basemodel['n']) * self.km['maxEllipseSize'][k]**2
                    
                    # If the actual measure if above the threshold, perform a
                    # "wait action" (if possible) to reduce it
                    if self.km[delta]['cov_tilde_measure'][k+delta] > self.km['maxEllipseSize'][k]:
                        # Reduce the uncertainty after taking the action, until it's below the threshold
                        
                        # Set waiting time equal to the worst-case possible
                        self.km[delta]['waiting_time'][k] = self.N
                        
                        # Determine how much stabilization time is needed to steer below the threshold
                        gamma = 0
                        delta_lowest = min(self.setup['deltas'])
                                                
                        # Copy the measure on the covariance matrix
                        covStabilize = self.km[delta]['cov'][k+delta]
                        covMeasure = self.km[delta]['cov_tilde_measure'][k+delta]
                        
                        flag = False
                        
                        while not flag and k + delta + gamma*delta_lowest < self.N:
                            if covMeasure <= self.km['maxEllipseSize'][k+gamma*delta_lowest]:
                                # If measure is now below threshold, we're okay
                                flag = True
                                if self.setup['verbose']:
                                    print('Have to wait',gamma,'steps of width',delta_lowest)
                                    print('Measure is reduced to',covMeasure)
                                
                                # Store how long we must wait to reduce uncertainty
                                self.km[delta]['waiting_time'][k] = copy.copy(gamma)
                                
                            else:
                                if self.setup['verbose']:
                                    print('Covariance measure too high at',covMeasure,'; gamma is',gamma)
                                
                                # Increment gamma value (stabilization time)
                                gamma += 1
                                
                                # Determine resulting covariance after wait
                                _, _, _, covStabilize, covMeasure = \
                                    kalmanFilter(self.model[delta_lowest], covStabilize)                            
                    
                # Print normal row in table
                if self.setup['verbose']:
                    tab.print_row([delta, k, 'Measure '+
                       str(np.round(self.km[delta]['cov_tilde_measure'][k+delta], decimals=3))])
                
            # Delete iterable variable
            del k
        del delta
        
        self.time['3_KalmanFilter'] = tocDiff(False)
        print('Kalman filter predictions done - time:',self.time['3_KalmanFilter'])
        
    def calculateTransitions(self):
        '''
        -----------------------------------------------------------------------
        CALCULATE TRANSITION PROBABILITIES
        -----------------------------------------------------------------------
        '''
           
        # Column widths for tabular prints
        col_width = [8,8,8,46]
        tab = table(col_width)
        
        # Determine minimum action time step width
        min_delta = min(self.setup['deltas'])
        
        self.abstr['prob'] = dict()
        
        if self.setup['sa']['switch']:
            
            # Create memory for scenario probabilities (6 variables/columns, max N+1 rows)
            # Used to ensure that equal probabilities are only calcualted once
            # If a file is specified, use it; otherwise define empty matrix
            self.abstr['memory'] = dict()
            
            if self.setup['sa']['usetable']:
                with open(self.setup['sa']['usetable'], newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for i,row in enumerate(reader):
                        if i != 0:
                            strSplit = row[0].split(',')
                            key = tuple( [int(i) for i in strSplit[0:3]] + [float(strSplit[3])] )
                            value = [float(i) for i in strSplit[-2:]]
                            self.abstr['memory'][key] = value
        
        # For every delta value in the list
        for delta_idx,delta in enumerate(self.setup['deltas']):
            self.abstr['prob'][delta] = dict()
            
            # For every time step in the horizon
            for k in range(self.N):
                self.abstr['prob'][delta][k] = dict()
            
                # (Re)initialize probability memory dictionary
                prob_mem = dict()
            
                # If LIC is active, transition probabilities are the
                # same for every delta value (only the waiting times
                # differ), so we can copy it from the first results
                if self.setup['LIC'] and delta > min_delta:

                    print(' -- Copy transition probabilities for',delta,'at time',k)
                    
                    # Copy transition probabilities
                    self.abstr['prob'][delta][k] = \
                        self.abstr['prob'][min_delta][k]
                        
                else:
                    
                    # Print header row
                    if self.setup['verbose']:
                        tab.print_row(['DELTA','K','ACTION','STATUS'], head=True)    
                    else:
                        print(' -- Calculate transitions for delta',delta,'at time',k)
            
                    # For every action (i.e. target state)
                    for a in range(self.abstr['nr_regions']):
                        
                        # Check if action a is available in any state at all
                        if len(self.abstr['enabledActions_inv'][delta][a]) > 0 \
                            or self.setup['LIC']:
                                
                            self.abstr['prob'][delta][k][a] = dict()
                        
                            if self.setup['sa']['switch']:
                                
                                # Calculate transition probability
                                self.abstr['memory'], \
                                self.abstr['prob'][delta][k][a] = \
                                    calculateTransitionProbabilities_SA(self.setup, self.abstr,
                                    self.basemodel['n'], a, 
                                    self.abstr['targetPoints'][a], self.km[delta]['cov_tilde'][k+delta])
                            
                            elif self.setup['efficientProbIntegral']:
                                
                                # Initialize probability list
                                self.abstr['prob'][delta][k][a]['approx'] = np.zeros(self.abstr['nr_regions']) #[0 for i in range(self.abstr['nr_regions'])]
                                
                                # For every possible resulting region
                                for j in range(self.abstr['nr_regions']):
                                    
                                    coord_distance = tuple(self.abstr['P'][a]['distances'][j])
                                    coord_distance_min = tuple(-self.abstr['P'][a]['distances'][j])
                                    
                                    # Check if we already calculated this one
                                    if coord_distance in prob_mem:
                                        
                                        # If so, copy from previous
                                        self.abstr['prob'][delta][k][a]['approx'][j] = prob_mem[coord_distance]
                                        
                                    elif coord_distance_min in prob_mem:
                                        
                                        # If so, copy from previous
                                        self.abstr['prob'][delta][k][a]['approx'][j] = prob_mem[coord_distance_min]
                                        
                                    else:
                                        
                                        # If not, calculate and add to dictionary
                                        '''
                                        self.abstr['prob'][delta][k][a]['approx'][j] = floor_decimal(mvn.mvnun(self.abstr['P'][j]['low'], 
                                            self.abstr['P'][j]['upp'], self.abstr['targetPoints'][a], 
                                            self.km[delta]['cov_tilde'][k+delta])[0], 4)
                                        '''
                                        self.abstr['prob'][delta][k][a]['approx'][j] = mvn.mvnun(self.abstr['P'][j]['low'], 
                                            self.abstr['P'][j]['upp'], self.abstr['targetPoints'][a], 
                                            self.km[delta]['cov_tilde'][k+delta])[0]
                                        
                                        prob_mem[coord_distance] = self.abstr['prob'][delta][k][a]['approx'][j]
                                
                            else:
                        
                                # Calculate transition probability
                                self.abstr['prob'][delta][k][a]['approx'] = \
                                    computeGaussianProbabilities(self.abstr['P'], 
                                    self.abstr['targetPoints'][a], self.km[delta]['cov_tilde'][k+delta])
                                                    
                            # Print normal row in table
                            if self.setup['verbose']:
                                if self.setup['sa']['switch']:
                                    prob_lb = np.round(max(self.abstr['prob'][delta][k][a]['lb']), 5)
                                    prob_ub = np.round(max(self.abstr['prob'][delta][k][a]['ub']), 5)
                                    tab.print_row([delta, k, a, 'sampling-based primary trans. prob bounds: ['+str(prob_lb)+', '+str(prob_ub)+']'])
                                
                                else:
                                    prob = np.round(max(self.abstr['prob'][delta][k][a]['approx']), 5)
                                    tab.print_row([delta, k, a, 'Gaussian-based approximated primary trans. prob.: '+str(prob)])
                            
                    # Delete iterable variables
                    del a
            del k
        del delta
        
        self.time['4_probabilities'] = tocDiff(False)
        print('Transition probabilities calculated - time:',self.time['4_probabilities'])

    def buildMDP(self):
        '''
        -----------------------------------------------------------------------
        CONSTRUCT MDP
        -----------------------------------------------------------------------
        '''
        
        self.mdp = createMDP(self.setup, self.N, self.abstr, self.km)
        
        self.time['5_MDPcreated'] = tocDiff(False)
        print('MDP created - time:',self.time['5_MDPcreated'])
            
    def solveMDP(self):
        '''
        -----------------------------------------------------------------------
        CONSTRUCT MDP
        -----------------------------------------------------------------------
        '''
        
        from .mainFunctions import valueIteration
        
        # Solve MDP via value iteration
        self.mdp['optimal_policy'], self.mdp['optimal_delta'], self.mdp['optimal_reward'] = \
            valueIteration(self.setup, self.mdp, self.N)
        
        # Convert result vectors to matrices
        self.results = dict()
        self.results['optimal_delta'] = np.reshape(self.mdp['optimal_delta'], (self.N, self.mdp['Sp']))
        self.results['optimal_policy'] = np.reshape(self.mdp['optimal_policy'], (self.N, self.mdp['Sp']))
        self.results['optimal_reward'] = np.reshape(self.mdp['optimal_reward'], (self.N, self.mdp['Sp']))
        
        self.plot           = dict()
    
        for delta_idx, delta in enumerate(self.setup['deltas']):
            self.plot[delta] = dict()
            
            self.plot[delta]['Nstart'] = 0
            self.plot[delta]['Nhalf']  = int(np.floor(self.N/2))-1
            self.plot[delta]['Nfinal'] = (self.N-1)- self.model[delta]['tau']
            
            # Convert index to absolute time (note: index 0 is time tau)
            self.plot[delta]['Tstart'] = int(self.plot[delta]['Nstart'] * self.basemodel['tau']) + self.basemodel['tau']
            self.plot[delta]['Thalf']  = int(self.plot[delta]['Nhalf'] * self.basemodel['tau']) + self.basemodel['tau']
            self.plot[delta]['Tfinal'] = int(self.plot[delta]['Nfinal'] * self.basemodel['tau']) + self.basemodel['tau']
           
        self.time['6_MDPsolved'] = tocDiff(False)
        print('MDP solved in',self.time['6_MDPsolved'])
        
    def generatePlots(self, delta_value, max_delta):
        '''
        -----------------------------------------------------------------------
        GENERATE PLOTS
        -----------------------------------------------------------------------
        '''
        
        print('\nGenerate plots')
        
        from .postprocessing.createPlots import createProbabilityPlots, createTimegroupingPlot
        
        if self.setup['probabilityPlots']:
            createProbabilityPlots(self.setup, self.plot[delta_value], self.N, self.model[delta_value], \
                                   self.results, self.abstr, self.mc)
                
            createTimegroupingPlot(self.setup, self.plot[delta_value], \
                                   self.basemodel, self.results, self.abstr, max_delta)
        
        toc()
        print('\nCode execution finished')
        
    def monteCarlo(self):
        '''
        -----------------------------------------------------------------------
        MONTE CARLO SIMULATION TO VALIDATE THE OBTAINED RESULTS
        -----------------------------------------------------------------------
        '''
        
        tocDiff(False)
        if self.setup['verbose']:
            print(' -- Starting Monte Carlo simulations...')
        
        self.mc['setup']                 = dict()
        self.mc['results']               = dict()
        self.mc['setup']['iterations']   = self.setup['MCiter']
        
        # Determine minimum action time step width
        min_delta = min(self.setup['deltas'])
        
        # For each time step to compare
        if self.setup['MC_kStart'] is False:
            self.mc['setup']['start_times'] = [self.plot[min_delta]['Nstart'],
                                                self.plot[min_delta]['Nhalf'],
                                                self.plot[min_delta]['Nfinal']]
        else:
            self.mc['setup']['start_times'] = self.setup['MC_kStart']
        
        # Check if we should limit the set of initial states for the MC sims.        
        if self.setup['MC_initS'] is not False:
            # Determine the index set of the initial states
            self.mc['initS'] = [ i for i,region in self.abstr['P'].items()
                if region['center'][0:(self.basemodel['n']-1)] == self.setup['MC_initS'] ]
            
        n_list = self.mc['setup']['start_times']
        
        self.mc['results']['reachability_probability'] = np.zeros((self.abstr['nr_regions'],len(n_list)))
        
        # Column widths for tabular prints
        if self.setup['verbose']:
            col_width = [6,8,6,6,46]
            tab = table(col_width)
        
        if self.setup['verbose']:
            print(' -- Computing required Gaussian random variables...')
        
        # The gaussian random variables are precomputed to speed up the code
        w_array = dict()
        for delta in self.setup['deltas']:
            w_array[delta] = np.random.multivariate_normal(self.model[delta]['noise']['w_mean'], self.model[delta]['noise']['w_cov'],
               ( len(n_list), self.abstr['nr_regions'], self.mc['setup']['iterations'], self.N ))
        
        v_array = np.random.multivariate_normal([0]*self.basemodel['r'], self.basemodel['noise']['v_cov'], 
               ( len(n_list), self.abstr['nr_regions'], self.mc['setup']['iterations'], self.N ))
        
        # For each starting time step in the list
        for n0id,n0 in enumerate(n_list):
            
            self.mc['results'][n0] = dict()
        
            # For each initial state
            for i,regionA in self.abstr['P'].items():
                
                # Check if we should indeed perform Monte Carlo sims for this state
                if self.setup['MC_initS'] is not False and i not in self.mc['initS']:
                    # If the current state is not in the list of initial states,
                    # continue to the next iteration
                    continue
                # Otherwise, continue with the Monte Carlo simulation
                
                if self.setup['verbose']:
                    tab.print_row(['K0','STATE','ITER','K','STATUS'], head=True)
                else:
                    print(' --- Monte Carlo for start time',n0,'and state',i)
                
                # Create dictionaries for results related to partition i
                self.mc['results'][n0][i]  = dict()
                self.mc['results'][n0][i]['goalReached'] = np.full(self.mc['setup']['iterations'], False, dtype=bool)
                    
                # For each of the monte carlo iterations
                for m in range(self.mc['setup']['iterations']):
                    
                    # Retreive the initial action time-grouping to be chosen
                    # (given by the optimal policy to the MDP)
                    delta = self.results['optimal_delta'][n0,i]
                    waiting_time = 0
                    
                    if i in self.abstr['goal']:
                        # If the initial state is already the goal state, succes
                        # Then abort the current iteration, as we have achieved the goal
                        self.mc['results'][n0][i]['goalReached'][m] = True
                        
                        if self.setup['verbose']:
                            tab.print_row([n0, i, m, n0, 'Initial state is goal state'], sort="Success")
                    elif delta == 0:
                        # If delta is zero, no policy is known, and reachability is zero
                        if self.setup['verbose']:
                            tab.print_row([n0, i, m, n0, 'No initial policy known, so abort'], sort="Warning")
                    else:
                        if self.setup['verbose']:
                            tab.print_row([n0, i, m, n0, 'Start Monte Carlo iteration'])
                                            
                        # Initialize the current simulation
                        
                        x = np.zeros((self.N, self.basemodel['n']))
                        y = np.zeros((self.N, self.basemodel['r']))
                        kalman_mu = [None]*self.N
                        mu_goal = [None]*self.N
                        mu_region = np.zeros(self.N).astype(int)
                        u = [None]*self.N
                        
                        actionToTake = np.zeros(self.N).astype(int)
                        deltaToTake = np.zeros(self.N).astype(int)
                        waiting_times = np.zeros(self.N).astype(int)
                        
                        # Set initial time
                        k = n0
                        
                        # Construct initial belief
                        kalman_mu[n0] = np.array(regionA['center'])
                        
                        # Determine initial state and measurement
                        w_init = np.random.multivariate_normal(\
                                [0]*self.basemodel['n'], self.km[delta]['cov'][n0])
                        v_init = v_array[n0id, i, m, k]
                        
                        # True state model dynamics
                        x[n0] = kalman_mu[n0] + w_init
                        y[n0] = self.model[delta]['C'] @ np.array(x[n0]) + v_init
                        
                        # For each time step in the finite time horizon
                        while k < self.N:
                            
                            if self.setup['verbose']:
                                tab.print_row([n0, i, m, k, 'New time step'])
                            
                            if k == n0:
                                region_likelihood_list = np.arange(self.mdp['Sp'])
                            else:
                                # Determine the likehood of being in a state,
                                # based on previous action taken
                                prev_k = k - delta - waiting_time
                                prev_s = mu_region[prev_k] + prev_k * self.mdp['Sp']
                                
                                prev_probs = self.mdp['Tp'][prev_k] \
                                   [prev_s][(delta, actionToTake[prev_k])]
                            
                                region_likelihood_list = np.array(prev_probs).argsort()[::-1]
                            
                            # Determine to what region the MDP state belongs   
                            # Default value is -1 (absorbing state)
                            mu_region[k] = -1
                            for ii in region_likelihood_list:
                                # If the current true state is in the set of polypoints..                                
                                if in_hull( kalman_mu[k], self.abstr['P'][ii]['hull'] ):
                                    # Save that state is currently in region ii
                                    mu_region[k] = ii
                                    
                                    # Retreive the action from the policy
                                    actionToTake[k] = self.results['optimal_policy'][k, mu_region[k]]
                                        
                                    # Print current state of the belief
                                    if self.setup['verbose']:
                                        tab.print_row([n0, i, m, k, 'Belief is in state '+str(mu_region[k])])
                                        
                                    break
                            
                            # If current region is the goal state ... 
                            if mu_region[k] in self.abstr['goal']:
                                # Then abort the current iteration, as we have achieved the goal
                                self.mc['results'][n0][i]['goalReached'][m] = True
                                
                                if self.setup['verbose']:
                                    tab.print_row([n0, i, m, k, 'Goal state reached'], sort="Success")
                                break
                            # If current region is in critical states...
                            elif mu_region[k] in self.abstr['critical']:
                                # Then abort current iteration
                                if self.setup['verbose']:
                                    tab.print_row([n0, i, m, k, 'Critical state reached, so abort'], sort="Warning")
                                break
                            elif mu_region[k] == -1:
                                if self.setup['verbose']:
                                    tab.print_row([n0, i, m, k, 'Absorbing state reached, so abort'], sort="Warning")
                                break
                            elif actionToTake[k] == 0:
                                if self.setup['verbose']:
                                    tab.print_row([n0, i, m, k, 'No policy known, so abort'], sort="Warning")
                                break
                            
                            # If loop was not aborted, we have a valid action
    
                            # Update the value of the time-grouping of the action
                            # dictated by the optimal policy
                            deltaToTake[k] = self.results['optimal_delta'][k, mu_region[k]]
                            delta = deltaToTake[k]
                            
                            # If delta is zero, no policy is known, and reachability is zero
                            if delta == 0:
                                if self.setup['verbose']:
                                    tab.print_row([n0, i, m, k, 'Action type undefined, so abort'], sort="Warning")
                                break
                            
                            if self.setup['LIC']:
                                # If the local information controller is active,
                                # also account for the "waiting time"
                                
                                # Retreive gamma value
                                gamma = self.km[delta]['waiting_time'][k]
                                
                                # Multiply with the minimum time step width to get waiting time
                                waiting_times[k] = gamma*min_delta
                                waiting_time = waiting_times[k]
                                
                                # Print message
                                if self.setup['verbose']:
                                    tab.print_row([n0, i, m, k, 'Wait for '+str(gamma)+'*'+\
                                               str(min_delta)+'='+str(waiting_time)+' time steps'])
                            else:
                                # If local information controller is not active,
                                # simply set the waiting time to zero
                                gamma = 0
                                waiting_times[k] = 0
                                waiting_time = waiting_times[k]
                            
                            if self.setup['verbose']:
                                tab.print_row([n0, i, m, k, 'Take action: '+str(actionToTake[k])+' (delta='+str(delta)+', gamma='+str(gamma)+')'])
                            
                            # Only perform another movement if k < N-tau (of base model)
                            if k < self.N - self.basemodel['tau']:
                            
                                # Move predicted mean to the future belief to the target point of the next state
                                mu_goal[k+delta] = self.abstr['targetPoints'][actionToTake[k]]
        
                                # Reconstruct the control input required to achieve this target point
                                # Note that we do not constrain the control input; we already know that a suitable control exists!
                                u[k] = np.array(self.model[delta]['B_pinv'] @ ( mu_goal[k+delta] - self.model[delta]['A'] @ kalman_mu[k] - self.model[delta]['noise']['w_mean'] ))
                                
                                # Implement the control into the physical (unobservable) system
                                x[k+delta] = self.model[delta]['A'] @ x[k] + self.model[delta]['B'] @ u[k] + w_array[delta][n0id, i, m, k]
                                y[k+delta] = self.model[delta]['C'] @ x[k+delta] + v_array[n0id, i, m, k]
                                
                                # Update the belief based on action and measurement                    
                                kalman_mu[k+delta] = mu_goal[k+delta] + self.km[delta]['K_gain'][k+delta] @ \
                                                        (y[k+delta] - self.model[delta]['C'] @ mu_goal[k+delta])
                                        
                                # Update the belief for the waiting time
                                for g in range(gamma):
                                    # Currently at time k, plus delta of action, plus waiting time
                                    kg = k + delta + (g+1)*min_delta
                                    
                                    # If waiting does not fit in time horizon anymore, break
                                    if kg >= self.N - self.basemodel['tau']:
                                        if self.setup['verbose']:
                                            tab.print_row([n0, i, m, kg-min_delta, 'Time horizon exceeded by waiting, so abort'], sort="Warning")
                                        break
                                    
                                    if self.setup['verbose']:
                                        tab.print_row([n0, i, m, kg-min_delta, 'Wait until time: '+str(kg)])
                                        
                                    # Reconstruct the control input required to achieve this target point
                                    # Note that we do not constrain the control input; we already know that a suitable control exists!
                                    u[kg] = np.array(self.model[min_delta]['B_pinv'] @ ( mu_goal[k+delta] - self.model[min_delta]['A'] @ kalman_mu[kg-min_delta] - self.model[delta]['noise']['w_mean'] ))
                                    
                                    # Since we wait in the same state, dynamics are changed accordingly
                                    x[kg] = self.model[min_delta]['A'] @ x[kg-min_delta] + self.model[min_delta]['B'] @ u[kg] + w_array[min_delta][n0id, i, m, k]
                                    y[kg] = self.model[min_delta]['C'] @ x[kg] + v_array[n0id, i, m, kg]
                                    
                                    # Update the belief based on action and measurement
                                    kalman_mu[kg] = mu_goal[k+delta] + self.km[min_delta]['K_gain'][kg] @ \
                                                            (y[kg] - self.model[min_delta]['C'] @ mu_goal[k+delta])
                        
                            # Increase iterator variable by the value of delta associated to chosen action
                            k += delta + waiting_time
                            
                        # One iteration complete                        
                        if self.setup['MCtrajectoryPlot'] and n0 == 0:
                            fID = "k0="+str(n0)+"_s0="+str(i)+"_iter="+str(m)
                            
                            plotTrajectory(setup=self.setup, fID=fID, k0=n0, k_end=k, 
                               x=x, y=y, actions=actionToTake, deltas=deltaToTake, 
                               waitings=waiting_times, mu_pred=mu_goal,
                               mu=kalman_mu, cov_tilde=self.km[min_delta]['cov_tilde'], 
                               cov=self.km[min_delta]['cov'], abstr=self.abstr)
                        
                # Set of monte carlo iterations completed for specific initial state
                
                # Calculate the overall reachability probability for initial state i
                self.mc['results']['reachability_probability'][i,n0id] = \
                    np.sum(self.mc['results'][n0][i]['goalReached']) / self.mc['setup']['iterations']
                    
        self.time['7_MonteCarlo'] = tocDiff(False)
        print('Monte Carlo simulations finished:',self.time['7_MonteCarlo'])