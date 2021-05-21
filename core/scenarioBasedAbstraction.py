# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:18:50 2021

@author: Thom Badings
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import copy
import csv
import sys
import os

from scipy.spatial import Delaunay

from .mainFunctions import definePartitions, defineDistances, \
    in_hull, makeModelFullyActuated, computeScenarioBounds, computeScenarioBounds_sparse, \
    computeRegionCenters
from .commons import tic, toc, ticDiff, tocDiff, \
    nearest_point_in_hull, table, printWarning
from .postprocessing.createPlots import createPartitionPlot

from .createMDP import mdp

'''
------------------------------------------------------------------------------
Main filter-based abstraction object definition
------------------------------------------------------------------------------
'''

from numba import njit

@njit
def f(mat, vec):
    return mat - vec

# @njit
# def f2(mat, threshold):
#     return np.maximum(np.max(mat, axis=1), -np.min(mat, axis=1)) <= threshold

@njit
def f2(mat, threshold):
    
    outVec = np.zeros(len(mat))
    for i,row in enumerate(mat):
        if max(np.abs(row)) <= threshold:
            outVec[i] = True
            
    return outVec

class Abstraction(object):
    '''
    Main abstraction object    
    '''
    
    def _defModel(self, delta):
        
        if delta == 0:
            model = makeModelFullyActuated(copy.deepcopy(self.basemodel), manualDimension = 'auto', observer=False)
        else:
            model = makeModelFullyActuated(copy.deepcopy(self.basemodel), manualDimension = delta, observer=False)
            
        # Determine inverse A matrix
        model.A_inv  = np.linalg.inv(model.A)
        
        # Determine pseudo-inverse B matrix
        model.B_pinv = np.linalg.pinv(model.B)
        
        # Retreive system dimensions
        model.p      = np.size(model.B,1)   # Nr of inputs
    
        # Control limitations
        model.uMin   = np.array( model.setup['control']['limits']['uMin'] * int(model.p/self.basemodel.p) )
        model.uMax   = np.array( model.setup['control']['limits']['uMax'] * int(model.p/self.basemodel.p) )
        
        return model
    
    def _defPartition(self):
        
        # Define partitioning of state-space
        abstr = dict()
        
        abstr['P'] = definePartitions(self.basemodel.n,
                                     self.basemodel.setup['partition']['nrPerDim'],
                                     self.basemodel.setup['partition']['width'],
                                     self.basemodel.setup['partition']['origin'],
                                     onlyCenter = False)
        
        abstr['nr_regions'] = len(abstr['P'])
        
        centerTuples = [tuple(abstr['P'][i]['center']) for i in range(abstr['nr_regions'])] 
        
        abstr['allCenters'] = dict(zip(centerTuples, range(abstr['nr_regions'])))
        
        # Determine goal regions
        abstr['goal'] = self._defStateLabelSet(abstr['allCenters'], 
            self.basemodel.setup['partition'], self.basemodel.setup['specification']['goal'])
        
        # Determine critical regions
        abstr['critical'] = self._defStateLabelSet(abstr['allCenters'], 
            self.basemodel.setup['partition'], self.basemodel.setup['specification']['critical'])
        
        return abstr
    
    def _defStateLabelSet(self, allCenters, partition, subset):
        
        if np.shape(subset)[1] == 0:
            return []
        
        else:
        
            # Retreive list of points and convert to array
            points = np.array( subset ) 
        
            # Compute all centers of regions associated with points
            centers = computeRegionCenters(points, partition)
            
            # Filter to only keep unique centers
            centers_unique = np.unique(centers, axis=0)

            # Return the ID's of regions associated with the unique centers            
            return [allCenters[tuple(c)] for c in centers_unique 
                               if tuple(c) in allCenters]
    
    def _defAllCorners(self):
            
        # Create all combinations of n bits, to reflect combinations of all lower/upper bounds of partitions
        bitCombinations = list(itertools.product([0, 1], repeat=self.basemodel.n))
        bitRelation     = ['low','upp']
        
        # Calculate all corner points of every partition.
        # Every partition has an upper and lower bounnd in every state (dimension).
        # Hence, the every partition has 2^n corners, with n the number of states.
        
        allOriginPointsNested = [[[
            self.abstr['P'][i][bitRelation[bit]][bitIndex] 
                for bitIndex,bit in enumerate(bitList)
            ] for bitList in bitCombinations 
            ] for i in self.abstr['P']
            ]
        
        return np.array(allOriginPointsNested)
    
    def _defRegionHull(self, points):
        
        return Delaunay(points, qhull_options='QJ')
    
    def _createTargetPoints(self, cornerPoints):
        
        target = dict()
        
        if self.basemodel.setup['targets']['nrPerDim'] != 'auto':

            # Width (span) between target points in every dimension
            if self.basemodel.setup['targets']['domain'] != 'auto':
                targetWidth = np.array(self.basemodel.setup['targets']['domain'])*2 / (self.basemodel.setup['targets']['nrPerDim'])
            else:
                targetWidth = np.array(self.basemodel.setup['partition']['nrPerDim']) * \
                    self.basemodel.setup['partition']['width'] / (self.basemodel.setup['targets']['nrPerDim'])
        
            # Create target points (similar to a partition of the state space)
            target['d'] = definePartitions(self.basemodel.n,
                    self.basemodel.setup['targets']['nrPerDim'],
                    targetWidth,
                    self.basemodel.setup['partition']['origin'],
                    onlyCenter = True)
        
        else:
        
            # Create target points of actions
            if len(self.abstr['goal']) == 1 and self.basemodel.setup['targets']['dynamic']:
            
                # Define the (unique) continuous goal state
                goal = self.abstr['P'][ self.abstr['goal'][0] ]['center'][0]
                    
                # Define dynamic target points
                target['d'] = [nearest_point_in_hull(cornerPoints[j], goal) for
                                                   j in range(self.abstr['nr_regions'])]
                    
            else:
                
                # Set default target points to the center of every region
                target['d'] = [self.abstr['P'][j]['center'] for j in range(self.abstr['nr_regions'])]
        
        targetPointTuples = [tuple(point) for point in target['d']]        
        target['inv'] = dict(zip(targetPointTuples, range(len(target['d']))))
        
        return target
    
    def _defInvArea(self, delta):
        
        ### Define inverse control area (note: not a hull yet)
        # Determine the set of extremal control inputs
        u = [[self.model[delta].uMin[i], self.model[delta].uMax[i]] for i in range(self.model[delta].p)]
        
        # Determine the inverse image of the extreme control inputs from target point
        x_inv_area = np.zeros((2**self.model[delta].p, self.model[delta].n))
        for i,elem in enumerate(itertools.product(*u)):
            list_elem = list(elem)
            
            # Calculate inverse image of the current extreme control input
            x_inv_area[i,:] = - self.model[delta].A_inv @ \
                (self.model[delta].B @ np.array(list_elem).T)  
    
        return x_inv_area
    
    def _defInvHull(self, x_inv_area):
        
        ### Define control hull
        x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
        
        return x_inv_hull
    
    def _defInvVec(self, delta):

        u_avg = np.array(self.model[delta].uMax + self.model[delta].uMin)/2    

        # Compute basis vectors (with average of all controls as origin)
        u = np.tile(u_avg, (self.basemodel.n,1)) + \
            np.diag(self.model[delta].uMax - u_avg)
        
        print(u)
        
        origin = - self.model[delta].A_inv @ \
                (self.model[delta].B @ np.array(u_avg).T)   
                
        basis_vectors = np.zeros((self.basemodel.n, self.basemodel.n))
        
        for i,elem in enumerate(u):
            # Calculate inverse image of the current extreme control input
            point = - self.model[delta].A_inv @ \
                (self.model[delta].B @ elem.T)    
            
            basis_vectors[i,:] = point - origin
        
        return basis_vectors
    
    def _defEnabledActions(self, delta):
            
        # Define dictionaries to sture points in the preimage of a state, and the
        # corresponding polytope points
        
        # Compute inverse reachability area
        x_inv_area = self._defInvArea(delta)
        
        total_actions_enabled = 0
        
        enabled_polypoints = dict()
        mu_inv_hull = dict()
        
        enabled_in_states = [[] for i in range(self.abstr['nr_actions'])]
        enabled_actions   = [[] for i in range(self.abstr['nr_regions'])]
        
        nr_corners = 2**self.basemodel.n
        
        printEvery = 1#min(100, max(1, int(self.abstr['nr_actions']/10)))
        
        # Check if dimension of control area equals that if the state vector
        dimEqual = self.model[delta].p == self.basemodel.n
        
        if dimEqual:
            
            print(' -- Computing inverse basis vectors...')
            # Use preferred method: map back the skewed image to squares
            
            basis_vectors = self._defInvVec(delta)   
            print('Basis vectors:',basis_vectors)
            
            basis_vectors_inv = np.linalg.inv( basis_vectors )
            print('Transformation:',basis_vectors_inv)
            
            print('Normal inverse area:',x_inv_area)
            
            x_inv_area_normalized = x_inv_area @ basis_vectors_inv
            print('Normalized hypercube:',x_inv_area_normalized)
            
            x_inv_area_offOrigin = np.average(x_inv_area_normalized, axis=0)
            print('Off origin:',x_inv_area_offOrigin)
            
            print('Shifted normalized hypercube:',x_inv_area @ basis_vectors_inv - x_inv_area_offOrigin)
            
            allCornersTransformed = self.abstr['allCornersFlat'] @ basis_vectors_inv - x_inv_area_offOrigin
            
        else:
            
            print(' -- Creating inverse hull for delta =',delta,'...')
            
            # Use standard method: check if points are in (skewed) hull
            x_inv_hull = self._defInvHull(x_inv_area)
            
            print('Normal inverse area:',x_inv_area)
        
            allCornersTransformed = self.abstr['allCornersFlat'] 
        
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        
        # For every action
        for action_id in range(self.abstr['nr_actions']):
            
            targetPoint = self.abstr['target']['d'][action_id]
            
            disturbance = self.model[delta].noise['w_mean']
            
            if dimEqual:
            
                # Shift the origin points (instead of the target point)
                originShift = self.model[delta].A_inv @ (np.array(targetPoint) - disturbance)
                
                # Python implementation
                originPointsShifted = allCornersTransformed - (originShift @ basis_vectors_inv)
                
                '''
                # Numba implementation
                originPointsShifted = f(allCornersTransformed, originShift)
                '''
                                
                # Reshape the whole matrix that we obtain
                poly_reshape = np.reshape( originPointsShifted,
                                (self.abstr['nr_regions'], nr_corners*self.basemodel.n))
                
                # Somehow, the line below is slower than the newer one below it.
                # enabled_in = np.max( abs(poly_reshape), axis=1 ) <= 1.0
                
                # Enabled actions are ones that have all corner points within
                # the origin-centered hypercube with unit length
                enabled_in = np.maximum(np.max(poly_reshape, axis=1), -np.min(poly_reshape, axis=1)) <= 1.0
                
                # enabled_any = sum( np.minimum(np.max(poly_reshape, axis=1), -np.min(poly_reshape, axis=1)) <= 1.0 )
                # print('Any point included in nr states:',enabled_any)
                
                '''
                # Numba implementation
                enabled_in = f2(poly_reshape, 1.0)
                '''
                
            else:
                
                # Shift the origin points (instead of the target point)
                originShift = self.model[delta].A_inv @ (np.array(targetPoint) - disturbance)
            
                # Subtract the shift from all corner points
                originPointsShifted = allCornersTransformed - originShift
            
                # Check which points are in the convex hull
                polypoints_vec = in_hull(originPointsShifted, x_inv_hull)
            
                # Map the enabled corner points of the partitions to actual partitions
                enabled_polypoints[action_id] = np.reshape(  polypoints_vec, 
                              (self.abstr['nr_regions'], nr_corners))
            
                # Polypoints contains the True/False results for all corner points of every partition
                # An action is enabled in a state if it is enabled in all its corner points
                enabled_in = np.all(enabled_polypoints[action_id] == True, axis = 1)
            
            # Shift the inverse hull to account for the specific target point
            if self.setup.plotting['partitionPlot'] and action_id == int(self.abstr['nr_regions']/2):

                predecessor_set = x_inv_area + np.tile(originShift, (nr_corners, 1))
            
            # Retreive the ID's of all states in which the action is enabled
            enabled_in_states[action_id] = np.nonzero(enabled_in)[0]
            
            # Remove critical states from the list of enabled actions
            enabled_in_states[action_id] = np.setdiff1d(
                enabled_in_states[action_id], self.abstr['critical'])
            
            if action_id % printEvery == 0:
                print(' -- Action',str(action_id),'enabled in',
                      str(len(enabled_in_states[action_id])),'states')
            
            if len(enabled_in_states[action_id]) > 0:
                total_actions_enabled += 1
            
            for origin in enabled_in_states[action_id]:
                enabled_actions[origin] += [action_id]
                
        enabledActions_inv = [enabled_in_states[i] 
                              for i in range(self.abstr['nr_actions'])]
        
        # Generate partition plot for the goal state, also showing the pre-image
        if self.setup.plotting['partitionPlot']:
            
            print('Create partition plot...')
            
            createPartitionPlot(self.abstr['goal'], delta, self.setup, \
                        self.model[delta], self.abstr, self.abstr['allCorners'],
                        predecessor_set)
        
        return total_actions_enabled, enabled_actions, enabledActions_inv
                
    def __init__(self):
        '''
        Initialize the scenario-based abstraction object.

        Parameters
        ----------
        setup : dict
        Setup dictionary.

        Returns
        -------
        None.

        '''
        
        print('Base model loaded:')
        print(' -- Dimension of the state:',self.basemodel.n)
        print(' -- Dimension of the control input:',self.basemodel.p)
        
        # Simulation end time is the end time
        self.T  = self.basemodel.setup['endTime']
        
        # Number of simulation time steps
        self.N = int(self.T) #/self.basemodel.tau)
        
        self.model = dict()
        for delta in self.setup.deltas:
            
            # Define model object for current delta value
            self.model[delta] = self._defModel(delta)
        
        self.time['0_init'] = tocDiff(False)
        print('Abstraction object initialized - time:',self.time['0_init'])
        
    def definePartition(self):
        ''' 
        -----------------------------------------------------------------------
        DEFINE THE DISCRETE STATE PARTITION
        -----------------------------------------------------------------------
        '''
        
        print('Computing partition of the state space...')
        
        self.abstr = self._defPartition()
        
        print(' -- Number of regions:',self.abstr['nr_regions'])
        print(' -- Goal regions are:',self.abstr['goal'])
        print(' -- Critical regions are:',self.abstr['critical'])

        self.time['1_partition'] = tocDiff(False)
        print('Discretized states defined - time:',self.time['1_partition'])

    def defineActions(self):
        ''' 
        -----------------------------------------------------------------------
        CREATE TARGET POINTS
        -----------------------------------------------------------------------
        '''
        
        print('Creating actions (target points)...')
        
        self.abstr['allCorners']     = self._defAllCorners()
        self.abstr['allCornersFlat'] = np.concatenate( self.abstr['allCorners'] )
        
        # Create the target point for every action (= every state)
        self.abstr['target'] = self._createTargetPoints(self.abstr['allCorners'])
        self.abstr['nr_actions'] = len(self.abstr['target']['d'])
        
        print(' -- Number of actions (target points):',self.abstr['nr_actions'])
        
        # Create hull of the full state space
        origin = np.array(self.basemodel.setup['partition']['origin'])
        halfWidth = np.array(self.basemodel.setup['partition']['nrPerDim']) \
            / 2 * self.basemodel.setup['partition']['width']
        stateSpaceBox = np.vstack((origin-halfWidth, origin+halfWidth))
        
        outerCorners = list( itertools.product(*stateSpaceBox.T) )
        
        print(' -- Creating hull for the full state space...')
        
        self.abstr['stateSpaceHull'] = self._defRegionHull(outerCorners)
        
        ''' 
        -----------------------------------------------------------------------
        DETERMINE WHICH ACTIONS ARE ACTUALLY ENABLED
        -----------------------------------------------------------------------
        '''
        
        print('Computing set of enabled actions...')
        
        self.abstr['actions']     = dict()
        self.abstr['actions_inv'] = dict()
        
        # For every time step grouping value in the list
        for delta_idx,delta in enumerate(self.setup.deltas):
            
            nr_A, self.abstr['actions'][delta], self.abstr['actions_inv'][delta] = \
                self._defEnabledActions(delta)
                        
            print(nr_A,'actions enabled')
            if nr_A == 0:
                printWarning('No actions enabled at all, so terminate')
                sys.exit()
        
        self.time['2_enabledActions'] = tocDiff(False)
        print('Enabled actions define - time:',self.time['2_enabledActions'])
        
###############################

class scenarioBasedAbstraction(Abstraction):
    def __init__(self, setup, basemodel):
        
        # Copy setup to internal variable
        self.setup = setup
        self.basemodel = basemodel
        
        # Define empty dictionary for monte carlo simulations (even if not used)
        self.mc = dict()   
        
        # Start timer
        tic()
        ticDiff()
        self.time = dict()
        
        Abstraction.__init__(self)
        
    def _loadScenarioTable(self, tableFile):
        
        # Create memory for scenario probabilities (6 variables/columns, max N+1 rows)
        # Used to ensure that equal probabilities are only calculated once
        # If a file is specified, use it; otherwise define empty matrix
        memory = dict()
        
        if not os.path.isfile(tableFile):
            sys.exit('ERROR: the following table file does not exist:'+str(tableFile))
        
        with open(tableFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i,row in enumerate(reader):
                
                strSplit = row[0].split(',')
                
                key = tuple( [int(float(i)) for i in strSplit[0:2]] + [float(strSplit[2])] )
                
                value = [float(i) for i in strSplit[-2:]]
                memory[key] = value
                    
        return memory
        
    def _computeProbabilityBounds(self, tab, k, delta):
        
        prob = dict()

        printEvery = min(10, max(1, int(self.abstr['nr_actions']/10)))

        # For every action (i.e. target point)
        for a in range(self.abstr['nr_actions']):
            
            # Check if action a is available in any state at all
            if len(self.abstr['actions_inv'][delta][a]) > 0:
                    
                prob[a] = dict()
            
                mu = self.abstr['target']['d'][a]
                Sigma = self.model[delta].noise['w_cov']
                samples = np.random.multivariate_normal(mu, Sigma, 
                                        size=self.setup.scenarios['samples'])
                    
                self.trans['memory'], prob[a] = \
                    computeScenarioBounds_sparse(self.setup, 
                      self.basemodel.setup['partition'], 
                      self.abstr, self.trans, samples)
                
                # Print normal row in table
                if a % printEvery == 0:
                    # prob_lb = np.round(max(prob[a]['lb']), 5)
                    # prob_ub = np.round(max(prob[a]['ub']), 5)
                    # tab.print_row([delta, k, a, 'sampling-based primary bounds: ['+str(prob_lb)+', '+str(prob_ub)+']'])
                    tab.print_row([delta, k, a, 'Probabilities computed'])
                
        return prob
    
    def defActions(self):
        Abstraction.definePartition(self)
        Abstraction.defineActions(self)
    
    def defTransitions(self):
        
        '''
        -----------------------------------------------------------------------
        CALCULATE TRANSITION PROBABILITIES
        -----------------------------------------------------------------------
        '''
           
        # Column widths for tabular prints
        col_width = [8,8,8,46]
        tab = table(col_width)
        
        self.trans = {'prob': {}}
                
        print(' -- Loading scenario approach table...')
        
        tableFile = 'input/probabilityTable_N='+ \
                        str(self.setup.scenarios['samples'])+'_beta='+ \
                        str(self.setup.scenarios['confidence'])+'.csv'
        
        # Load scenario approach table
        self.trans['memory'] = self._loadScenarioTable(tableFile = tableFile)
        
        # Retreive type of horizon
        if self.setup.mdp['horizon'] == 'finite':
            k_range = range(self.N)
        else:
            k_range = [0]
        
        print('Computing transition probabilities...')
        
        # For every delta value in the list
        for delta_idx,delta in enumerate(self.setup.deltas):
            self.trans['prob'][delta] = dict()
            
            # For every time step in the horizon
            for k in k_range:
                
                # Print header row
                if self.setup.main['verbose']:
                    tab.print_row(['DELTA','K','ACTION','STATUS'], head=True)    
                else:
                    print(' -- Calculate transitions for delta',delta,'at time',k)
                
                self.trans['prob'][delta][k] = self._computeProbabilityBounds(tab, k, delta)
                
            # Delete iterable variables
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
        
        # Initialize MDP object
        self.mdp = mdp(self.setup, self.N, self.abstr)
        
        if self.setup.mdp['solver'] == 'Python':
            self.mdp.createPython(self.abstr, self.trans)
            
        elif self.setup.mdp['solver'] == 'PRISM':
            
            if self.setup.mdp['prism_model_writer'] == 'explicit' and self.setup.mdp['horizon'] != 'finite':
                
                # Create PRISM file (explicit way)
                self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
                    self.mdp.writePRISM_explicit(self.abstr, self.trans, self.setup.mdp['mode'])   
            
            else:
            
                # Create PRISM file (default way)
                self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
                    self.mdp.writePRISM_scenario(self.abstr, self.trans, self.setup.mdp['mode'], self.setup.mdp['horizon'])   

        self.time['5_MDPcreated'] = tocDiff(False)
        print('MDP created - time:',self.time['5_MDPcreated'])
            
    def solveMDP(self):
        
        # Solve MDP, either internally or via PRISM
        if self.setup.mdp['solver'] == 'Python':
        
            # Solve the MDP using the internal solver
            self._solveMDPviaPython()
            
        elif self.setup.mdp['solver'] == 'PRISM':
            
            # Solve the MDP in PRISM (which is called via the terminal)
            self._solveMDPviaPRISM()
        
        # Process results
        self.plot           = dict()
    
        for delta_idx, delta in enumerate(self.setup.deltas):
            self.plot[delta] = dict()
            self.plot[delta]['N'] = dict()
            self.plot[delta]['T'] = dict()
            
            self.plot[delta]['N']['start'] = 0
            
            # Convert index to absolute time (note: index 0 is time tau)
            self.plot[delta]['T']['start'] = int(self.plot[delta]['N']['start'] * self.basemodel.tau)
            
            if self.setup.mdp['solver'] == 'Python':
                self.plot[delta]['N']['half']  = int(np.floor(self.N/2))-1
                self.plot[delta]['N']['final'] = (self.N-1)- self.model[delta].tau
                
                # Convert index to absolute time (note: index 0 is time tau)
                self.plot[delta]['T']['half']  = int(self.plot[delta]['N']['half'] * self.basemodel.tau)
                self.plot[delta]['T']['final'] = int(self.plot[delta]['N']['final'] * self.basemodel.tau)
           
        self.time['6_MDPsolved'] = tocDiff(False)
        print('MDP solved in',self.time['6_MDPsolved'])
        
    def _solveMDPviaPython(self):
        '''
        -----------------------------------------------------------------------
        CONSTRUCT MDP
        -----------------------------------------------------------------------
        '''
        
        from .mainFunctions import valueIteration

        # Solve MDP via value iteration        
        solution = valueIteration(self.setup, self.mdp)
        
        # Convert result vectors to matrices
        self.results = dict()
        
        self.results['optimal_policy'] = np.reshape(solution['optimal_policy'], (self.N, self.mdp.nr_regions))
        self.results['optimal_delta'] = np.reshape(solution['optimal_delta'], (self.N, self.mdp.nr_regions))
        self.results['optimal_reward'] = np.reshape(solution['optimal_reward'], (self.N, self.mdp.nr_regions))
        
    def _solveMDPviaPRISM(self):
        
        import subprocess
        import pandas as pd

        prism_folder = self.setup.mdp['prism_folder'] 
        
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
        print('Starting PRISM...')
        
        spec = self.mdp.specification
        mode = self.setup.mdp['mode']
        java_memory = self.setup.mdp['prism_java_memory']
        
        print(' -- Running PRISM with specification for mode',mode.upper()+'...')
    
        file_prefix     = self.setup.directories['outputFcase'] + "PRISM_" + mode
        policy_file     = file_prefix + '_policy.csv'
        vector_file     = file_prefix + '_vector.csv'
    
        options = ' -ex -exportadv "'+policy_file+'"'+ \
                  ' -exportvector "'+vector_file+'"'
    
        # Switch between PRISM command for explicit model vs. default model
        if self.setup.mdp['prism_model_writer'] == 'explicit' and self.setup.mdp['horizon'] != 'finite':
    
            print(' --- Execute PRISM command for EXPLICIT model description')        
    
            model_file      = '"'+self.mdp.prism_file+'"'        
            spec_file       = '"'+self.mdp.spec_file+'"'        
        
            # Explicit model
            command = prism_folder+"bin/prism -javamaxmem "+str(java_memory)+"g "+ \
                      "-importmodel "+model_file+" -pf '"+spec+"' "+options
        else:
            
            print(' --- Execute PRISM command for DEFAULT model description')
            
            model_file      = '"'+self.mdp.prism_file+'"'
            
            # Default model
            command = prism_folder+"bin/prism -javamaxmem "+str(java_memory)+"g "+ \
                      model_file+" -pf '"+spec+"' "+options    
        
        subprocess.Popen(command, shell=True).wait()    
            
        self.results = dict()
        
        policy_all = pd.read_csv(policy_file, header=None).iloc[:, 1:].fillna(-1).to_numpy()
        # Flip policy upside down (PRISM generates last time step at top!)
        policy_all = np.flipud(policy_all)
        
        self.results['optimal_policy'] = np.zeros(np.shape(policy_all))
        self.results['optimal_delta'] = np.zeros(np.shape(policy_all))
        self.results['optimal_reward'] = np.zeros(np.shape(policy_all))
        
        rewards_k0 = pd.read_csv(vector_file, header=None).iloc[1:].to_numpy()
        self.results['optimal_reward'][0,:] = rewards_k0.flatten()
        
        # Split the optimal policy between delta and action itself
        for i,row in enumerate(policy_all):
            
            for j,value in enumerate(row):
                
                # If value is not -1 (means no action defined)
                if value != -1:
                    # Split string
                    value_split = value.split('_')
                    # Store action and delta value separately
                    self.results['optimal_policy'][i,j] = int(value_split[1])
                    self.results['optimal_delta'][i,j] = int(value_split[3])
                else:
                    # If no policy is known, set to -1
                    self.results['optimal_policy'][i,j] = int(value)
                    self.results['optimal_delta'][i,j] = int(value) 
        
    def generatePlots(self, delta_value, max_delta):
        '''
        -----------------------------------------------------------------------
        GENERATE PLOTS
        -----------------------------------------------------------------------
        '''
        
        print('\nGenerate plots')
        
        from .postprocessing.createPlots import createProbabilityPlots, createTimegroupingPlot
        
        if self.setup.plotting['probabilityPlots']:
            createProbabilityPlots(self.setup, self.plot[delta_value], self.N, self.model[delta_value], \
                                   self.results, self.abstr, self.mc)
                
        toc()
        
    def monteCarlo(self, iterations='auto', init_states='auto', init_times='auto'):
        '''
        -----------------------------------------------------------------------
        MONTE CARLO SIMULATION TO VALIDATE THE OBTAINED RESULTS
        -----------------------------------------------------------------------
        '''
        
        tocDiff(False)
        if self.setup.main['verbose']:
            print(' -- Starting Monte Carlo simulations...')
        
        self.mc['results']  = dict()
        self.mc['traces']   = dict()
        
        if iterations != 'auto':
            self.setup.montecarlo['iterations'] = int(iterations)
        if init_states != 'auto':
            self.setup.montecarlo['init_states'] = list(init_states)
        if init_times != 'auto':
            self.setup.montecarlo['init_timesteps'] = list(init_times)
        elif self.setup.montecarlo['init_timesteps'] is False:
            # Determine minimum action time step width
            min_delta = min(self.setup.deltas)
            
            self.setup.montecarlo['init_timesteps'] = [ n for n in self.plot[min_delta]['N'].values() ]
        
        # Check if we should limit the set of initial states for the MC sims.        
        # if self.setup.montecarlo['init_states'] is not False:
        #     # Determine the index set of the initial states
        #     self.mc['initS'] = [ i for i,region in self.abstr['P'].items()
        #         if region['center'][0:(self.basemodel.n-1)] == \
        #             self.setup.montecarlo['init_states'] ]
            
        n_list = self.setup.montecarlo['init_timesteps']
        
        # print(' -- Creating hulls for all regions...')
        
        # Create hull for every region in the partition
        # if 'hull' not in self.abstr['P'][0]:
        #     for i in self.abstr['P']:
        #         self.abstr['P'][i]['hull'] = self._defRegionHull(self.abstr['allCorners'][i])
        
        self.mc['results']['reachability_probability'] = \
            np.zeros((self.abstr['nr_regions'],len(n_list)))
        
        # Column widths for tabular prints
        if self.setup.main['verbose']:
            col_width = [6,8,6,6,46]
            tab = table(col_width)

            print(' -- Computing required Gaussian random variables...')
        
        if self.setup.montecarlo['init_states'] == False:
            init_state_idxs = np.arange(self.abstr['nr_regions'])
            
        else:
            init_state_idxs = self.setup.montecarlo['init_states']
        
        # The gaussian random variables are precomputed to speed up the code
        w_array = dict()
        for delta in self.setup.deltas:
            w_array[delta] = np.random.multivariate_normal(self.model[delta].noise['w_mean'], self.model[delta].noise['w_cov'],
               ( len(n_list), len(init_state_idxs), self.setup.montecarlo['iterations'], self.N ))
        
        # For each starting time step in the list
        for n0id,n0 in enumerate(n_list):
            
            self.mc['results'][n0] = dict()
            self.mc['traces'][n0] = dict()
        
            # For each initial state
            for i_abs,i in enumerate(init_state_idxs):
                
                regionA = self.abstr['P'][i]
                
                # Check if we should indeed perform Monte Carlo sims for this state
                if self.setup.montecarlo['init_states'] is not False and i not in self.setup.montecarlo['init_states']:
                    print('cannot happen!')
                    # If the current state is not in the list of initial states,
                    # continue to the next iteration
                    continue
                # Otherwise, continue with the Monte Carlo simulation
                
                if self.setup.main['verbose']:
                    tab.print_row(['K0','STATE','ITER','K','STATUS'], head=True)
                else:
                    print(' -- Monte Carlo for start time',n0,'and state',i)
                
                # Create dictionaries for results related to partition i
                self.mc['results'][n0][i]  = dict()
                self.mc['results'][n0][i]['goalReached'] = np.full(self.setup.montecarlo['iterations'], False, dtype=bool)
                self.mc['traces'][n0][i]  = dict()    
                
                # For each of the monte carlo iterations
                for m in range(self.setup.montecarlo['iterations']):
                    
                    self.mc['traces'][n0][i][m] = []
                    
                    # Retreive the initial action time-grouping to be chosen
                    # (given by the optimal policy to the MDP)
                    delta = self.results['optimal_delta'][n0,i]
                    
                    if i in self.abstr['goal']:
                        # If the initial state is already the goal state, succes
                        # Then abort the current iteration, as we have achieved the goal
                        self.mc['results'][n0][i]['goalReached'][m] = True
                        
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 'Initial state is goal state'], sort="Success")
                    elif delta == 0:
                        # If delta is zero, no policy is known, and reachability is zero
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 'No initial policy known, so abort'], sort="Warning")
                    else:
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 'Start Monte Carlo iteration'])
                                            
                        # Initialize the current simulation
                        
                        x = np.zeros((self.N, self.basemodel.n))
                        x_goal = [None]*self.N
                        x_region = np.zeros(self.N).astype(int)
                        u = [None]*self.N
                        
                        actionToTake = np.zeros(self.N).astype(int)
                        deltaToTake = np.zeros(self.N).astype(int)
                        
                        # Set initial time
                        k = n0
                        
                        # True state model dynamics
                        x[n0] = np.array(regionA['center'])
                        
                        # Add current state to trace
                        self.mc['traces'][n0][i][m] += [x[n0]]
                        
                        # For each time step in the finite time horizon
                        while k < self.N / min(self.setup.deltas):
                            
                            if self.setup.main['verbose']:
                                tab.print_row([n0, i, m, k, 'New time step'])
                            
                            # if k == n0 or self.setup.mdp['solver'] != 'Python':
                            #     region_likelihood_list = np.arange(self.mdp.nr_regions)
                            # else:
                            #     # Determine the likehood of being in a state,
                            #     # based on previous action taken
                            #     prev_k = k - delta
                            #     prev_s = x_region[prev_k] + prev_k * self.mdp.nr_regions
                                
                            #     prev_probs = self.mdp.transitions['prob'][prev_k] \
                            #        [prev_s][(delta, actionToTake[prev_k])]
                            
                            #     region_likelihood_list = np.array(prev_probs).argsort()[::-1]
                            
                            # Determine to what region the MDP state belongs   
                            # Default value is -1 (absorbing state)
                            
                            # x_region[k] = -1
                            
                            # for ii in region_likelihood_list:
                            #     # If the current true state is in the set of polypoints..                                
                            #     if in_hull( x[k], self.abstr['P'][ii]['hull'] ):
                            #         # Save that state is currently in region ii
                            #         x_region[k] = ii
                                    
                            #         # Retreive the action from the policy
                            #         actionToTake[k] = self.results['optimal_policy'][k, x_region[k]]
                                        
                            #         # Print current state of the belief
                            #         if self.setup.main['verbose']:
                            #             tab.print_row([n0, i, m, k, 'True state is in region '+str(x_region[k])])
                                        
                            #         break
                                
                            ### Method below doesn't use convex hulls (= faster)
                            
                            # Determine to what region the MDP state belongs   
                            # Default value is -1 (absorbing state)
                            
                            # Compute all centers of regions associated with points
                            center_coord = computeRegionCenters(x[k], self.basemodel.setup['partition']).flatten()
                            
                            if tuple(center_coord) in self.abstr['allCenters']:
                                # Save that state is currently in region ii
                                x_region[k] = self.abstr['allCenters'][tuple(center_coord)]
                                
                                # Retreive the action from the policy
                                actionToTake[k] = self.results['optimal_policy'][k, x_region[k]]
                            else:
                                x_region[k] = -1
                            
                            ###
                            
                            # If current region is the goal state ... 
                            if x_region[k] in self.abstr['goal']:
                                # Then abort the current iteration, as we have achieved the goal
                                self.mc['results'][n0][i]['goalReached'][m] = True
                                
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Goal state reached'], sort="Success")
                                break
                            # If current region is in critical states...
                            elif x_region[k] in self.abstr['critical']:
                                # Then abort current iteration
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Critical state reached, so abort'], sort="Warning")
                                break
                            elif x_region[k] == -1:
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Absorbing state reached, so abort'], sort="Warning")
                                break
                            elif actionToTake[k] == -1:
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'No policy known, so abort'], sort="Warning")
                                break
                            
                            # If loop was not aborted, we have a valid action
    
                            # Update the value of the time-grouping of the action
                            # dictated by the optimal policy
                            deltaToTake[k] = self.results['optimal_delta'][k, x_region[k]]
                            delta = deltaToTake[k]
                            
                            # If delta is zero, no policy is known, and reachability is zero
                            if delta == 0:
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Action type undefined, so abort'], sort="Warning")
                                break
                            
                            if self.setup.main['verbose']:
                                tab.print_row([n0, i, m, k, 'In state: '+str(x_region[k])+', take action: '+str(actionToTake[k])+' (delta='+str(delta)+')'])
                            
                            # Only perform another movement if k < N-tau (of base model)
                            if k < self.N:
                            
                                # Move predicted mean to the future belief to the target point of the next state
                                x_goal[k+delta] = self.abstr['target']['d'][actionToTake[k]]
        
                                # Reconstruct the control input required to achieve this target point
                                # Note that we do not constrain the control input; we already know that a suitable control exists!
                                u[k] = np.array(self.model[delta].B_pinv @ ( x_goal[k+delta] - self.model[delta].A @ x[k] - self.model[delta].noise['w_mean'] ))
                                
                                # Implement the control into the physical (unobservable) system
                                x_hat = self.model[delta].A @ x[k] + self.model[delta].B @ u[k]
                                
                                x[k+delta] = x_hat + w_array[delta][n0id, i_abs, m, k]
                            
                                # Add current state to trace
                                self.mc['traces'][n0][i][m] += [x[k+delta]]
                            
                            # Increase iterator variable by the value of delta associated to chosen action
                            k += delta
                        
                # Set of monte carlo iterations completed for specific initial state
                
                # Calculate the overall reachability probability for initial state i
                self.mc['results']['reachability_probability'][i,n0id] = \
                    np.sum(self.mc['results'][n0][i]['goalReached']) / self.setup.montecarlo['iterations']
                    
        self.time['7_MonteCarlo'] = tocDiff(False)
        print('Monte Carlo simulations finished:',self.time['7_MonteCarlo'])