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
import itertools                # Import to crate iterators
import copy                     # Import to copy variables in Python
import sys                      # Allows to terminate the code at sorme point
from scipy.spatial import Delaunay, ConvexHull # Import to create convex hulls
from scipy.spatial.qhull import _Qhull

from .mainFunctions import definePartitions, computeRegionOverlap, \
    in_hull, makeModelFullyActuated, cubic2skew, skew2cubic, point_in_hull
from .commons import tocDiff, printWarning, findMinDiff, tic, ticDiff, \
    extractRowBlockDiag
from .postprocessing.createPlots import partitionPlot2D, partitionPlot3D

from .createMDP import mdp

'''
------------------------------------------------------------------------------
Main filter-based abstraction object definition
------------------------------------------------------------------------------
'''

class Abstraction(object):
    '''
    Main abstraction object    
    '''
    
    def __init__(self):
        '''
        Initialize the scenario-based abstraction object.

        Returns
        -------
        None.

        '''
        
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
        # Define empty dictionary for monte carlo sims. (even if not used)
        self.mc = dict()   
        
        # Start timer
        tic()
        ticDiff()
        self.time = dict()
        
        print('Base model loaded:')
        print(' -- Dimension of the state:',self.system.LTI['n'])
        print(' -- Dimension of the control input:',self.system.LTI['p'])
        
        # Number of simulation time steps
        self.N = int(self.system.endTime)
        
        # Reduce step size as much as possible
        base_delta = self.system.base_delta
        self.setup.divide = 1
        
        for div in range(base_delta, 0, -1):
            if base_delta % div == 0:
                
                print(' -- Simplify step size by a factor of',div)
                self.N = int(self.N / div)
                self.system.base_delta  = (np.array(self.system.base_delta) / div).astype(int)
                self.setup.divide = div
                
                break
            
        self.setup.jump_deltas = self.system.base_delta * self.system.adaptive['rates']
        self.setup.all_deltas = np.concatenate(([self.system.base_delta], 
                                                self.setup.jump_deltas)).astype(int)
        
        self.model = dict()
        for delta in self.setup.all_deltas:
            
            # Define model object for current delta value
            self.model[delta] = self._defModel(int(delta*div))
        
        self.time['0_init'] = tocDiff(False)
        print('Abstraction object initialized - time:',self.time['0_init'])
    
    def _defModel(self, delta):
        '''
        Define model within abstraction object for given value of delta

        Parameters
        ----------
        delta : int
            Value of delta to create model for.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        
        if delta == 0:
            model = makeModelFullyActuated(copy.deepcopy(self.system.LTI), 
                       manualDimension = 'auto', observer=False)
        else:
            model = makeModelFullyActuated(copy.deepcopy(self.system.LTI), 
                       manualDimension = delta, observer=False)
            
        # Determine inverse A matrix
        model['A_inv']  = np.linalg.inv(model['A'])
        
        # Determine pseudo-inverse B matrix
        model['B_pinv'] = np.linalg.pinv(model['B'])
        
        # Retreive system dimensions
        model['p']      = np.size(model['B'],1)   # Nr of inputs
    
        # Control limitations
        model['uMin']   = np.array( self.system.control['limits']['uMin'] * \
                                int(model['p']/self.system.LTI['p']) )
        model['uMax']   = np.array( self.system.control['limits']['uMax'] * \
                                int(model['p']/self.system.LTI['p']) )
        
        # If noise samples are used, recompute them
        if self.setup.scenarios['gaussian'] is False:
            
            model['noise']['samples'] = np.vstack(
                (2*model['noise']['samples'][:,0],
                 model['noise']['samples'][:,0],
                 2*model['noise']['samples'][:,1],
                 model['noise']['samples'][:,1],
                 2*model['noise']['samples'][:,2],
                 model['noise']['samples'][:,2])).T
        
        uAvg = (model['uMin'] + model['uMax']) / 2
        
        if np.linalg.matrix_rank(np.eye(model['n']) - model['A']) == model['n']:
            model['equilibrium'] = np.linalg.inv(np.eye(model['n']) - model['A']) @ \
                (model['B'] @ uAvg + model['Q_flat'])
        
        return model
    
    def _defPartition(self):
        
        # Define partitioning of state-space
        abstr = dict()
        
        abstr['P'] = definePartitions(self.system.LTI['n'],
                            self.system.partition['nrPerDim'],
                            self.system.partition['width'],
                            self.system.partition['origin'],
                            onlyCenter = False)
        
        abstr['nr_regions'] = len(abstr['P'])
        abstr['origin'] = self.system.partition['origin']
        
        centerTuples = [tuple(np.round(region['center'], self.setup.precision)) 
                              for region in abstr['P'].values()] 
        
        abstr['allCentersCubic'] = dict(zip(centerTuples, 
                                       abstr['P'].keys()))
        
        return abstr
    
    def _defStateLabelSet(self, allVertices, blocks, typ="goal", epsilon=0):
        '''
        Returns the indices of regions associated with the unique centers.

        Parameters
        ----------
        allVertices : List
            List of the center coordinates for all regions.
        subset : List
            List of points to return the unique centers for.

        Returns
        -------
        list
            List of unique center points.

        '''
        
        if len(blocks) == 0:
            return {}
        
        else:
            
            limits_overlap = {}
            
            for bl,block in blocks.items():
                
                # Account for maximum error bound (epsilon)
                limits = block['limits']
                
                if typ == "goal":
                    
                    goalAnywhere = False
                    
                    # A region is a goal region if it is completely
                    # contained in the specified region (we want to have
                    # a convervative under-approximation)
                    
                    limits_eps = np.vstack((limits[:,0] + epsilon, limits[:,1] - epsilon)).T
                    
                elif typ == "critical":
                    
                    # A region is a critical region if it contains any
                    # part of the specified region (we want to have a
                    # convervative over-approximation)
                    
                    limits_eps = np.vstack((limits[:,0] - epsilon, limits[:,1] + epsilon)).T
                    
                if any(limits[:,1] - limits[:,0] <= 0):
                    return {}
                    
                for region,vertices in enumerate(allVertices):
                    
                    if all([    any(lims[0] < vertices[:,row]) 
                            and any(lims[1] > vertices[:,row]) 
                            for row,lims in enumerate(limits_eps) ]):
                        
                        limits_region = np.vstack((self.abstr['P'][region]['low'],
                                                  self.abstr['P'][region]['upp'])).T
                        
                        if region not in limits_overlap:
                            limits_overlap[region] = {}
                        
                        limits_overlap[region][bl] = computeRegionOverlap(limits_region, limits_eps)
                        
                        if typ == "goal" and limits_overlap[region][bl] is not None:
                            goalAnywhere = True
                        
        if typ == "goal" and goalAnywhere is False:
            printWarning('Warning: the augmented goal region is not contained in any region (epsilon='+str(epsilon)+')')
        
        return limits_overlap
    
    def _defAllVertices(self):
        '''
        Returns the vertices of every region in the partition (as nested list)

        Returns
        -------
        list
            Nested list of all vertices of all regions.

        '''
            
        # Create all combinations of n bits, to reflect combinations of all 
        # lower/upper bounds of partitions
        bitCombinations = list(itertools.product([0, 1], repeat=self.system.LTI['n']))
        bitRelation     = ['low','upp']
        
        # Calculate all corner points of every partition.
        # Every partition has an upper and lower bounnd in every state (dimension).
        # Hence, the every partition has 2^n corners, with n the number of states.
        
        allOriginPointsNested = [[[
            region[bitRelation[bit]][bitIndex] 
                for bitIndex,bit in enumerate(bitList)
            ] for bitList in bitCombinations 
            ] for region in self.abstr['P'].values()
            ]
        
        return np.array(allOriginPointsNested)
    
    def _defRegionHull(self, points):
        '''
        Compute the convex hull for the given points

        Parameters
        ----------
        points : 2D Numpy array
            Numpy array, with every row being a point to include in the hull.

        Returns
        -------
        Convex hull
            Convex hull object.

        '''
        
        return Delaunay(points, qhull_options='QJ')
    
    def _stateInCritical(self, point):
    
        if any([in_hull( point, block['hull'] ) for block in self.system.spec['critical'].values()]):
            return True
        else:
            return False
    
    def _createTargetPoints(self, delta):
        '''
        Create target points, based on the vertices given

        Returns
        -------
        target : dict
            Dictionary with all data about the target points.

        '''
        
        print(' ---- Create actions for rate delta =',delta)
        
        target = dict()
        
        if delta != self.system.base_delta:
            
            # If not at base delta, use manual list of target points
            # (for the time being)
            
            target['d'] = self.system.adaptive['target_points']
        
            '''
            target['d'] = [region['center'] for region in self.abstr['P'].values() 
                           if not self._stateInCritical(region['center']) ]
            '''
        
        elif self.system.targets['nrPerDim'] != 'auto':

            # Width (span) between target points in every dimension
            if self.system.targets['domain'] != 'auto':
                targetWidth = np.array(self.system.targets['domain'])*2 / (self.system.targets['nrPerDim'])
            else:
                targetWidth = np.array(self.system.partition['nrPerDim']) * \
                    self.system.partition['width'] / (self.system.targets['nrPerDim'])
        
            # Create target points (similar to a partition of the state space)
            target['d'] = definePartitions(self.system.LTI['n'],
                    self.system.targets['nrPerDim'],
                    targetWidth,
                    self.system.partition['origin'],
                    onlyCenter = True)
            
            # Transformed from cubic to skewed
            target['d'] = cubic2skew(target['d'])
        
        else:
        
            # Set default target points to the center of every region
            target['d'] = [region['center'] for region in self.abstr['P'].values() 
                           if not self._stateInCritical(region['center']) ] #+ \
                #[np.array([0,-.1, 0.01, 0.01])]
        
        targetPointTuples = [tuple(point) for point in target['d']]        
        target['inv'] = dict(zip(targetPointTuples, range(len(target['d']))))
        
        return target
    
    def _defInvArea(self, delta):
        '''
        Compute the predecessor set (without the shift due to the target
        point). This acccounts to computing, for all u_k, the set
        A^-1 (B u_k - q_k)

        Parameters
        ----------
        delta : int
            Delta value of the model which is used.

        Returns
        -------
        x_inv_area : 2D Numpy array
            Predecessor set (every row is a vertex).

        '''
        
        # Determine the set of extremal control inputs
        u = [[self.model[delta]['uMin'][i], self.model[delta]['uMax'][i]] for i in 
              range(self.model[delta]['p'])]
        
        # Determine the inverse image of the extreme control inputs
        x_inv_area = np.zeros((2**self.model[delta]['p'], self.model[delta]['n']))
        for i,elem in enumerate(itertools.product(*u)):
            list_elem = list(elem)
            
            # Calculate inverse image of the current extreme control input
            x_inv_area[i,:] = self.model[delta]['A_inv'] @ \
                (self.model[delta]['B'] @ np.array(list_elem).T + 
                 self.model[delta]['Q_flat'])  
    
        return x_inv_area
    
    def _defInvHull(self, x_inv_area):
        '''
        Define the convex hull object of the predecessor set        

        Parameters
        ----------
        x_inv_area : 2D Numpy array
            Predecessor set (every row is a vertex).

        Returns
        -------
        x_inv_hull : hull
            Convex hull object.

        '''
        
        x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
        
        return x_inv_hull
    
    def _defBasisVectors(self, delta):
        '''
        Compute the basis vectors of the predecessor set, computed from the
        average control inputs to the maximum in every dimension of the
        control space.
        Note that the drift does not play a role here.  

        Parameters
        ----------
        delta : int
            Delta value of the model which is used.

        Returns
        -------
        basis_vectors : 2D Numpy array
            Numpy array of basis vectors (every row is a vector).

        '''
        
        u_avg = np.array(self.model[delta]['uMax'] + self.model[delta]['uMin'])/2    

        # Compute basis vectors (with average of all controls as origin)
        u = np.tile(u_avg, (self.system.LTI['n'],1)) + \
            np.diag(self.model[delta]['uMax'] - u_avg)
        
        origin = self.model[delta]['A_inv'] @ \
                (self.model[delta]['B'] @ np.array(u_avg).T)   
                
        basis_vectors = np.zeros((self.system.LTI['n'], self.system.LTI['n']))
        
        for i,elem in enumerate(u):
            
            # Calculate inverse image of the current extreme control input
            point = self.model[delta]['A_inv'] @ \
                (self.model[delta]['B'] @ elem.T)    
            
            basis_vectors[i,:] = point - origin
        
            print(' ---- Length of basis',i,':',
                  np.linalg.norm(basis_vectors[i,:]))
        
        return basis_vectors
    
    def _defEnabledActions(self, delta, verbose=False):
        '''
        Define dictionaries to sture points in the preimage of a state, and
        the corresponding polytope points.

        Parameters
        ----------
        delta : int
            Delta value of the model which is used.

        Returns
        -------
        total_actions_enabled : int
            Total number of enabled actions.
        enabled_actions : list
            Nested list of actions enabled in every region.
        enabledActions_inv : list
            Nested list of inverse actions enabled in every region.
            
        '''
        
        from .commons import angle_between
        
        def f7(seq):
            seen = set()
            seen_add = seen.add
            return [int(x) for x in seq if not (x in seen or seen_add(x))]
        
        # Compute inverse reachability area
        x_inv_area = self._defInvArea(delta)
        
        total_actions_enabled = 0
        
        enabled_polypoints = dict()
        
        enabled_in_states = [[] for i in range(self.abstr['nr_actions'][delta])]
        enabled_actions   = [[] for i in range(self.abstr['nr_regions'])]
        
        nr_corners = 2**self.system.LTI['n']
        
        printEvery = min(100, max(1, int(self.abstr['nr_actions'][delta]/10)))
        
        # Check if dimension of control area equals that if the state vector
        dimEqual = self.model[delta]['p'] == self.system.LTI['n']
        
        if dimEqual:
            
            print('\nCompute enabled actions for delta='+str(delta)+' via linear transformation...')
            # Use preferred method: map back the skewed image to squares
            
            basis_vectors = self._defBasisVectors(delta)   
            
            if verbose:
                print(' >> Basis vectors of backward reachable set (BRS):\n',basis_vectors)
            
                for i,v1 in enumerate(basis_vectors):
                    for j,v2 in enumerate(basis_vectors):
                        if i != j:
                            print(' ---- Angle between control',i,'and',j,':',
                                  angle_between(v1,v2) / np.pi * 180)
            
            parralelo2cube = np.linalg.inv( basis_vectors )
            
            x_inv_area_normalized = x_inv_area @ parralelo2cube
            
            predSet_originShift = -np.average(x_inv_area_normalized, axis=0)
            
            if verbose:
                print('Shift of BRS from origin:',predSet_originShift)
            
            allRegionVertices = self.abstr['allVerticesFlat'] @ parralelo2cube \
                    - predSet_originShift
            
        else:
            
            print('\nCompute enabled actions for delta='+str(delta)+' based on convex hull...')
            
            # Use standard method: check if points are in (skewed) hull
            x_inv_hull = self._defInvHull(x_inv_area)
            
            x_inv_qHull = _Qhull(b"i", x_inv_area,
                  options=b"QJ",
                  furthest_site=False,
                  incremental=False, 
                  interior_point=None)
        
            allRegionVertices = self.abstr['allVerticesFlat'] 
        
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        
        # Put goal regions up front of the list of actions
        '''
        action_range = f7(np.concatenate(( list(self.abstr['goal']['X'][0].keys()),
                           np.arange(self.abstr['nr_actions']) )))
        '''
        
        action_range = np.arange(self.abstr['nr_actions'][delta]) 
        
        print('\nStart to compute the set of enabled actions for delta='+str(delta)+'...')
        
        # For every action
        for action_id in action_range:
            
            targetPoint = self.abstr['target'][delta]['d'][action_id]
            
            # If system is 'robot', than only enable jump actions if the velocity
            # is zero
            '''
            if self.system.name == 'double_integrator' and delta != 1 and targetPoint[1] != 0:
                continue
            
            # If system is UAV and local information controllers are enabled
            if self.system.name == 'UAV' and delta != 1:
                # Skip if any velocity component is notzero (because the 
                # required waiting will not be enabled anyways)
                # if self.system.LTI['n'] == 4 and any(targetPoint[[1,3]] != 0) or \
                #    self.system.LTI['n'] == 6 and any(targetPoint[[1,3,5]] != 0):
                #        continue
                   
                if action_id not in [772, 1002, 1682, 1692, 1477, 1027, 577]:
                    continue
            '''
            
            if dimEqual:
            
                # Shift the origin points (instead of the target point)
                A_inv_d = self.model[delta]['A_inv'] @ np.array(targetPoint)
                
                # Python implementation
                allVerticesNormalized = (A_inv_d @ parralelo2cube) - \
                                         allRegionVertices
                                
                # Reshape the whole matrix that we obtain
                poly_reshape = np.reshape( allVerticesNormalized,
                                (self.abstr['nr_regions'], 
                                 nr_corners*self.system.LTI['n']))
                
                # Enabled actions are ones that have all corner points within
                # the origin-centered hypercube with unit length
                enabled_in = np.maximum(np.max(poly_reshape, axis=1), 
                                        -np.min(poly_reshape, axis=1)) <= 1.0
                
            else:
                
                # Shift the origin points (instead of the target point)
                A_inv_d = self.model[delta]['A_inv'] @ np.array(targetPoint)
            
                # Subtract the shift from all corner points
                allVertices = A_inv_d - allRegionVertices
            
                # Check which points are in the convex hull
                polypoints_vec = in_hull(allVertices, x_inv_hull)
                polypoints_vec = point_in_hull(allVertices, x_inv_qHull)
            
                # Map enabled vertices of the partitions to actual partitions
                enabled_polypoints[action_id] = np.reshape(  polypoints_vec, 
                              (self.abstr['nr_regions'], nr_corners))
            
                # Polypoints contains the True/False results for all vertices
                # of every partition. An action is enabled in a state if it 
                # is enabled in all its vertices
                enabled_in = np.all(enabled_polypoints[action_id] == True, 
                                    axis = 1)
            
            # Shift the inverse hull to account for the specific target point
            if self.setup.plotting['partitionPlot'] and \
                action_id == int(self.abstr['nr_regions']/2):

                predecessor_set = A_inv_d - x_inv_area
                
                # Partition plot for the goal state, also showing pre-image
                print('Create partition plot...')
                
                partitionPlot2D((0,1), (2,3), self.abstr['goal']['X'][0], 
                    delta, self.setup, self.model[delta], self.system.partition, self.abstr, 
                    self.abstr['allVertices'], predecessor_set)
            
                # If number of dimensions is 3, create 3D plot
                if self.system.LTI['n'] == 3:
                    
                    partitionPlot3D(self.setup, self.model[delta], 
                                    self.abstr['allVertices'], predecessor_set)
            
            # Retreive the ID's of all states in which the action is enabled
            enabled_in_states[action_id] = np.nonzero(enabled_in)[0]
                
            if action_id % printEvery == 0 or delta > 1:
                if action_id in self.abstr['goal']['X'][0]:
                    print(' -- GOAL action',str(action_id),'enabled in',
                          str(len(enabled_in_states[action_id])),
                          'states - target point:',
                          str(targetPoint))
                    
                else:
                    print(' -- Action',str(action_id),'enabled in',
                          str(len(enabled_in_states[action_id])),
                          'states - target point:',
                          str(targetPoint))
            
            if len(enabled_in_states[action_id]) > 0:
                total_actions_enabled += 1
            
            for origin in enabled_in_states[action_id]:
                enabled_actions[origin] += [action_id]
                
        enabledActions_inv = [enabled_in_states[i] 
                              for i in range(self.abstr['nr_actions'][delta])]
        
        return total_actions_enabled, enabled_actions, enabledActions_inv
        
    def definePartition(self):
        ''' 
        Define the discrete state space partition and target points
        
        Returns
        -------
        None.
        '''
        
        # Create partition
        print('\nComputing partition of the state space...')
        
        self.abstr = self._defPartition()

        self.time['1_partition'] = tocDiff(False)
        print('Discretized states defined - time:',self.time['1_partition'])
        
        # Create target points
        print('\nCreating actions (target points)...')
        
        self.abstr['allVertices']     = self._defAllVertices()
        
        ##########
        
        if self.setup.main['skewed']:
        
            basis_vectors = self._defBasisVectors(max(self.setup.all_deltas)) * 2
            
            length = np.linalg.norm(basis_vectors, axis=1)
            normalization_factor = min(self.system.partition['width'] / length)
            
            # Normalize basis vectors
            self.abstr['basis_vectors'] = normalization_factor * basis_vectors

            
            for i in range(self.abstr['nr_regions']):
                
                # Compute skewed center region
                self.abstr['P'][i]['center'] = cubic2skew(self.abstr['P'][i]['center'], self.abstr)
                
                # Compute skewed vertices of region
                self.abstr['allVertices'][i,:,:] = cubic2skew(self.abstr['allVertices'][i], self.abstr)
        
        else:
            
            # If skewed partition disabled, just define identity matrix
            self.abstr['basis_vectors'] = np.eye(self.system.LTI['n'])
            
        self.abstr['basis_vectors_inv'] = np.linalg.inv( self.abstr['basis_vectors'] )
        
        # Find biggest cube that fits within a skewed region
        self.abstr['biggest_cube'] = np.array([findMinDiff(self.abstr['allVertices'][0][:,dim]) 
                                               for dim in range(self.system.LTI['n'])])
        
        ##########
        
        # Create flat (2D) array of all vertices
        self.abstr['allVerticesFlat'] = np.concatenate(self.abstr['allVertices'])
        
        self.abstr['goal'] = {'X': {}}
        self.abstr['critical'] = {'X': {}}
        
        for k in range(1,self.N + 1):    
            
            if self.setup.main['mode'] == 'Filter':
                epsilon = self.km[1][k]['error_bound']
            else:
                epsilon = np.zeros(self.system.LTI['n'])
        
            # Determine goal regions
            self.abstr['goal']['X'][k] = self._defStateLabelSet(
                self.abstr['allVertices'],
                self.system.spec['goal'], typ="goal", epsilon=epsilon)
            
            # Determine critical regions
            self.abstr['critical']['X'][k] = self._defStateLabelSet(
                self.abstr['allVertices'],
                self.system.spec['critical'], typ="critical", epsilon=epsilon)
           
        self.abstr['goal']['X'][0] = self.abstr['goal']['X'][1]
        self.abstr['critical']['X'][0] = self.abstr['critical']['X'][1]
        
        for delta in self.setup.jump_deltas:
            # Compute goal and critical regions for higher delta states
            
            self.abstr['goal'][delta] = dict()
            self.abstr['critical'][delta] = dict()
            
            for gamma in range(0, self.km['waiting_time'] + 1):
            
                if self.setup.main['mode'] == 'Filter':
                    epsilon = self.km[delta][gamma]['error_bound']
                else:
                    epsilon = np.zeros(self.system.LTI['n'])
            
                # Determine goal regions
                self.abstr['goal'][delta][gamma] = self._defStateLabelSet(
                    self.abstr['allVertices'],
                    self.system.spec['goal'], typ="goal", epsilon=epsilon)
                
                # Determine critical regions
                self.abstr['critical'][delta][gamma] = self._defStateLabelSet(
                    self.abstr['allVertices'],
                    self.system.spec['critical'], typ="critical", epsilon=epsilon)
            
        # Compute goal and critical regions without augmented bounds (for plotting)
        self.abstr['goal']['zero_bound'] = self._defStateLabelSet(
                self.abstr['allVertices'],
                self.system.spec['goal'], typ="goal", epsilon=0)
        self.abstr['critical']['zero_bound'] = self._defStateLabelSet(
                self.abstr['allVertices'],
                self.system.spec['critical'], typ="critical", epsilon=0)
        
        print(' -- Number of regions:',self.abstr['nr_regions'])
        print(' -- Number of goal regions:',len(self.abstr['goal']['zero_bound']))
        print(' -- Number of critical regions:',len(self.abstr['critical']['zero_bound']))
        
        # Create the target point for every action (= every state)
        self.abstr['target'] = {delta: self._createTargetPoints(delta) for 
                                delta in self.setup.all_deltas}
        
        self.abstr['nr_actions'] = {delta: len(self.abstr['target'][delta]['d']) for
                                    delta in self.setup.all_deltas}
        
        print(' -- Number of actions (target points):',self.abstr['nr_actions'])
        
    def defineActions(self):
        ''' 
        Determine which actions are actually enabled.
        
        Returns
        -------
        None.
        '''
        
        self.abstr['actions']     = dict()
        self.abstr['actions_inv'] = dict()
        
        nr_A_total = 0
        
        # For every time step grouping value in the list
        for delta in self.setup.all_deltas:
            
            nr_A, self.abstr['actions'][delta], \
             self.abstr['actions_inv'][delta] = self._defEnabledActions(delta)
                        
            print(nr_A,'actions enabled')
            
            nr_A_total += nr_A
            
        if nr_A_total == 0:
            printWarning('No actions enabled at all, so terminate')
            sys.exit()
    
        self.time['2_enabledActions'] = tocDiff(False)
        print('\nEnabled actions define - time:',self.time['2_enabledActions'])
        
    def buildMDP(self):
        '''
        Build the (i)MDP and create all respective PRISM files.

        Returns
        -------
        model_size : dict
            Dictionary describing the number of states, choices, and 
            transitions.

        '''
        
        # Initialize MDP object
        self.mdp = mdp(self.setup, self.N, self.abstr)
        
        if self.setup.mdp['prism_model_writer'] == 'explicit':
            
            # Create PRISM file (explicit way)
            model_size, self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
                self.mdp.writePRISM_explicit(self.abstr, self.trans, mode=self.setup.mdp['mode'], km=self.km)   
        
        else:
        
            # Create PRISM file (default way)
            self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
                self.mdp.writePRISM_scenario(self.abstr, self.trans, self.setup.mdp['mode'])  
              
            model_size = {'States':None,'Choices':None,'Transitions':None}

        self.time['4_MDPcreated'] = tocDiff(False)
        print('\ninterval MDP created - time:',self.time['4_MDPcreated'])
        
        return model_size
        
    def solveMDP(self):
        '''
        Solve the (i)MDP usign PRISM

        Returns
        -------
        None.

        '''
            
        # Solve the MDP in PRISM (which is called via the terminal)
        policy_file, vector_file = self._solveMDPviaPRISM()
        
        # Load PRISM results back into Python
        self.loadPRISMresults(policy_file, vector_file)
            
        self.time['5_MDPsolved'] = tocDiff(False)
        print('MDP solved in',self.time['5_MDPsolved'])
        
    def _solveMDPviaPRISM(self):
        '''
        Call PRISM to solve (i)MDP while executing the Python codes.

        Returns
        -------
        policy_file : str
            Name of the file in which the optimal policy is stored.
        vector_file : str
            Name of the file in which the optimal rewards are stored.

        '''
        
        import subprocess

        prism_folder = self.setup.mdp['prism_folder'] 
        
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
        print('Starting PRISM...')
        
        spec = self.mdp.specification
        mode = self.setup.mdp['mode']
        java_memory = self.setup.mdp['prism_java_memory']
        
        print(' -- Running PRISM with specification for mode',
              mode.upper()+'...')
    
        file_prefix = self.setup.directories['outputFcase'] + "PRISM_" + mode
        policy_file = file_prefix + '_policy.csv'
        vector_file = file_prefix + '_vector.csv'
    
        options = ' -ex -exportadv "'+policy_file+'"'+ \
                  ' -exportvector "'+vector_file+'"'
    
        # Switch between PRISM command for explicit model vs. default model
        if self.setup.mdp['prism_model_writer'] == 'explicit':
    
            print(' --- Execute PRISM command for EXPLICIT model description')        
    
            model_file      = '"'+self.mdp.prism_file+'"'             
        
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
        
        return policy_file, vector_file
                
    def loadPRISMresults(self, policy_file, vector_file):
        '''
        Load results from existing PRISM output files.

        Parameters
        ----------
        policy_file : str
            Name of the file to load the optimal policy from.
        vector_file : str
            Name of the file to load the optimal policy from.

        Returns
        -------
        None.

        '''
        
        def _split_policy(value):
            
            if value == 'step':
                # If action is to wait one step (for jump delta)
                return [-2, -2, -2]
                
            elif value != -1:
                # Split string
                values_split = value.split('_')
                # Store action and delta value separately
                return [int(value_split) for value_split in values_split]
                
            else:
                # If no policy is known, set to -1
                return [-1, -1, -1]
        
        import pandas as pd
        
        ovh = self.mdp.overhead
        
        policy_all = pd.read_csv(policy_file, header=None).iloc[:, ovh:].fillna(-1).to_numpy()
        
        # Flip policy upside down (PRISM generates last time step at top!)
        policy_all = np.flipud(policy_all)
        
        self.mdp.opt_reward = pd.read_csv(vector_file, header=None).iloc[
                ovh:ovh+self.mdp.nr_regions].to_numpy().flatten()
        
        #####
        
        # Compensate in the optimal reachability probabability for the prob.
        # that we already started in a goal/critical state. This is on itself
        # not modeled in the iMDP, so we must do it here. Note that this has no
        # effect on the optimal policy.
        
        pI_goal = self.trans['prob'][1][0]['prob_goal']
        pI_crit = self.trans['prob'][1][0]['prob_critical']
        
        self.mdp.opt_reward = pI_goal + (1-pI_goal-pI_crit)*self.mdp.opt_reward
        
        #####
        
        # Initialize policy columns to dataframe
        cols = ['opt_action', 'opt_delta', 'opt_ksucc_id']
        for col in cols:
            self.mdp.MAIN_DF[col] = -1
            self.mdp.MAIN_DF[col] = self.mdp.MAIN_DF[col].astype(object)
        
        # Loop over every row of the dataframe
        for index, row in self.mdp.MAIN_DF.iterrows():
            
            # Create policy matrix (every row is a time step)
            split_matrix = np.array([ _split_policy(value) for value in policy_all[:, index] ]).T
            
            self.mdp.MAIN_DF['opt_action'].loc[index]   = split_matrix[0]
            self.mdp.MAIN_DF['opt_delta'].loc[index]    = split_matrix[1]
            self.mdp.MAIN_DF['opt_ksucc_id'].loc[index] = split_matrix[2]
            
                    
    def generatePlots(self, delta_value, max_delta, case_id, writer,
                      iterative_results=None):
        '''
        Generate (optimal reachability probability) plots

        Parameters
        ----------
        delta_value : int
            Value of delta for the model to plot for.
        max_delta : int
            Maximum value of delta used for any model.

        Returns
        -------
        None.
        '''
        
        print('\nGenerate plots')
        
        performance = None
        
        if len(self.abstr['P']) <= 1000:
        
            from .postprocessing.createPlots import createProbabilityPlots
            
            if self.setup.plotting['probabilityPlots']:
                createProbabilityPlots(self.setup, self.model[delta_value]['n'],
                                       self.system.partition,
                                       self.mdp, self.abstr, self.mc)
                    
        else:
            
            printWarning("Omit probability plots (nr. of regions too large)")
            
        # The code below plots the trajectories
        if self.system.name in ['UAV', 'shuttle']:
            
            from core.postprocessing.createPlots import trajectoryPlot
        
            # Create trajectory plot
            performance_df, _ = trajectoryPlot(self, case_id, writer)
            
            if self.setup.main['iterative'] is True and iterative_results != None:
                performance = pd.concat(
                    [iterative_results['performance'], performance_df], axis=0)
    
        # The code below plots the heat map
        if self.system.name in ['building_1room','building_2room','double_integrator','UAV'] or \
            (self.system.name == 'UAV' and self.system.modelDim == 2):
            
            from core.postprocessing.createPlots import reachabilityHeatMap
            
            # Create heat map
            reachabilityHeatMap(self)
            
        return performance