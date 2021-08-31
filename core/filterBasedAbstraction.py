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
import csv                      # Import to create/load CSV files
import sys                      # Allows to terminate the code at some point
import os                       # Import OS to allow creationg of folders
import random                   # Import to use random variables
from scipy.stats import mvn, norm

from .mainFunctions import defineDistances, \
    computeRegionCenters, cubic2skew, skew2cubic, kalmanFilter, in_hull, \
    steadystateCovariance, steadystateCovariance_sdp, covarianceEllipseSize, \
    minimumEpsilon
from .commons import tic, ticDiff, tocDiff, table, printWarning, \
    floor_decimal, extractRowBlockDiag, is_pos_def

from .abstraction import Abstraction

class filterBasedAbstraction(Abstraction):
    def __init__(self, setup, system):
        '''
        Initialize filter-based abstraction (FiAb) object

        Parameters
        ----------
        setup : dict
            Setup dictionary.
        system : dict
            Base model for which to create the abstraction.

        Returns
        -------
        None.

        '''
        
        # Copy setup to internal variable
        self.setup = setup
        self.system = system
        self.km = dict()
        
        Abstraction.__init__(self)
        
        # Apply Kalman filter to compute covariance over time
        self.KalmanPrediction()
        
        # Define state space partition
        Abstraction.definePartition(self)
        
        # Compute array of all region centers
        allCentersArray = np.array([region['center'] for region in self.abstr['P'].values()])
        
        for a in range(self.abstr['nr_actions']):
            
            self.abstr['P'][a]['distances'] = self.abstr['target']['d'][a] - allCentersArray
            
            
    def _computeProbabilityBounds(self, tab, Sigma_worst, Sigma_best, actions_inv, GOAL, CRITICAL, verbose=True):
        '''
        Compute transition probability intervals (bounds)

        Parameters
        ----------
        

        Returns
        -------
        prob : dict
            Dictionary containing the computed transition probabilities.

        '''

        SigmaEqual = np.array_equal(Sigma_worst, Sigma_best)

        printEvery = min(100, max(1, int(self.abstr['nr_actions']/10)))

        nr_decimals = 5 # Number of decimals to round off on
        threshold_decimals = 4 # Minimum probability to model
                
        # Transform covariance back to hypercubic partition
        if self.setup.main['skewed']:
            SigmaCubic_worst = self.abstr['basis_vectors_inv'].T @ Sigma_worst @ self.abstr['basis_vectors_inv'] 
            SigmaCubic_best  = self.abstr['basis_vectors_inv'].T @ Sigma_best @ self.abstr['basis_vectors_inv'] 
        else:
            SigmaCubic_worst = Sigma_worst
            SigmaCubic_best  = Sigma_best
        
        ######
        
        if self.setup.main['covarianceMode'] == 'SDP':
            func = steadystateCovariance_sdp
        else:
            func = steadystateCovariance
        
        maxCov = func([SigmaCubic_worst, SigmaCubic_best], verbose=False)['worst']            
        beta = 10**-(threshold_decimals+1)
        
        limit_norm = minimumEpsilon( cov=maxCov, beta=beta, stepSize=0.1, singleParam = False ) + \
                        0.5*np.array(self.system.partition['width'])
        
        # (Re)initialize probability memory dictionary
        prob = dict()
        prob_worst_mem = dict()
        prob_best_mem  = dict()
        
        # For every action (i.e. target point)
        for a in range(self.abstr['nr_actions']):
            
            skipped_counter = 0
        
            # Check if action a is available in any state at all
            if len(actions_inv[a]) > 0:
                    
                prob[a] = dict()
                    
                # Retrieve and transform mean of distribution
                mu = self.abstr['target']['d'][a]
                
                if self.setup.main['skewed']:
                    muCubic = skew2cubic(mu, self.abstr)
                else:
                    muCubic = mu
                
                probs_worst_all = np.zeros(len(self.abstr['P']))
                probs_best_all  = np.zeros(len(self.abstr['P']))
                
                prob_goal_worst_sum = 0
                prob_goal_best_sum = 0
                
                prob_critical_worst_sum = 0
                prob_critical_best_sum = 0
                
                # For every possible resulting region
                for j in self.abstr['P'].keys():
                    
                    ### 1) Main transition probability
                    # Compute the vector difference between the target point
                    # and the current region
                    coord_distance = self.abstr['P'][a]['distances'][j]
                    
                    if any(np.abs(coord_distance) > limit_norm):
                        skipped_counter += 1
                        continue
                    
                    region = self.abstr['P'][j]
                    
                    # Check if we already calculated this one
                    if tuple(coord_distance) in prob_worst_mem and not self.setup.main['skewed']:
                        
                        # If so, copy from previous
                        probs_worst = prob_worst_mem[tuple(coord_distance)]
                        probs_best  = prob_best_mem[tuple(coord_distance)]
                        
                    elif tuple(-coord_distance) in prob_worst_mem and not self.setup.main['skewed']:
                        
                        # If so, copy from previous
                        probs_worst = prob_worst_mem[tuple(-coord_distance)]
                        probs_best  = prob_best_mem[tuple(-coord_distance)]
                        
                    else:    
                
                        # If both covariances are equal, only compute once
                        if SigmaEqual:
                            probs_worst = probs_best = \
                                floor_decimal(mvn.mvnun(
                                region['low'], region['upp'],
                                muCubic, SigmaCubic_worst)[0], threshold_decimals)
                            
                        else:
                            probs_worst = floor_decimal(mvn.mvnun(
                                region['low'], region['upp'],
                                muCubic, SigmaCubic_worst)[0], threshold_decimals)
                            
                            probs_best = floor_decimal(mvn.mvnun(
                                region['low'], region['upp'],
                                muCubic, SigmaCubic_best)[0], threshold_decimals)
                        
                        if not self.setup.main['skewed']:
                            prob_worst_mem[tuple(coord_distance)] = probs_worst
                            prob_best_mem[tuple(coord_distance)]  = probs_best
                    
                    ### 2) Probability to reach goal state
                    # Subtract the probability to end up in a goal state
                    if j in GOAL and (probs_worst > 0 or probs_best > 0):
                        
                        all_limits = GOAL[j]
                    
                        # If both covariances are equal, only compute once
                        if SigmaEqual:
                            prob_goal_worst = prob_goal_best = \
                                floor_decimal(sum([mvn.mvnun(
                                 list(lims[:,0]), list(lims[:,1]), muCubic, SigmaCubic_worst)[0] 
                                 for lims in all_limits.values()
                                 ]), 6)     
                    
                        else:
                            prob_goal_worst = floor_decimal(sum([mvn.mvnun(
                                list(lims[:,0]), list(lims[:,1]), muCubic, SigmaCubic_worst)[0] 
                                for lims in all_limits.values()
                                ]), 6)
                            prob_goal_best = floor_decimal(sum([mvn.mvnun(
                                list(lims[:,0]), list(lims[:,1]), muCubic, SigmaCubic_best)[0] 
                                for lims in all_limits.values()
                                ]), 6)
                            
                    else:
                        prob_goal_worst = 0
                        prob_goal_best  = 0
                    
                    ### 3) Probability to reach critical state
                    # Subtract the probability to end up in a critical state
                    if j in CRITICAL and (probs_worst > 0 or probs_best > 0):
                        
                        all_limits = CRITICAL[j]
                            
                        # If both covariances are equal, only compute once
                        if SigmaEqual:
                            prob_critical_worst = prob_critical_best = \
                                floor_decimal(sum([mvn.mvnun(
                                lims[:,0], lims[:,1], muCubic, SigmaCubic_worst)[0] 
                                for lims in all_limits.values()
                                ]), threshold_decimals)
                        
                        else:
                            prob_critical_worst = floor_decimal(sum([mvn.mvnun(
                                lims[:,0], lims[:,1], muCubic, SigmaCubic_worst)[0] 
                                for lims in all_limits.values()
                                ]), threshold_decimals)
                            
                            prob_critical_best = floor_decimal(sum([mvn.mvnun(
                                lims[:,0], lims[:,1], muCubic, SigmaCubic_best)[0] 
                                for lims in all_limits.values()
                                ]), threshold_decimals)
                        
                    else:
                        prob_critical_worst = 0
                        prob_critical_best  = 0
                        
                    probs_worst_all[j] = (probs_worst - prob_critical_worst - prob_goal_worst)
                    probs_best_all[j]  = (probs_best  - prob_critical_best  - prob_goal_best)
                    
                    prob_goal_worst_sum += prob_goal_worst
                    prob_goal_best_sum  += prob_goal_best
                
                    prob_critical_worst_sum += prob_critical_worst
                    prob_critical_best_sum += prob_critical_best                
                
                nonzero_idxs = np.array([idx for idx,(ub1,ub2) in enumerate(zip(probs_worst_all, probs_best_all)) 
                                         if ub1 > 0 or ub2 > 0])
                
                if len(nonzero_idxs) > 0:
                    approx = np.vstack((probs_worst_all[nonzero_idxs], probs_best_all[nonzero_idxs]))
                else:
                    approx = np.array([[]])
                goal        = np.array([prob_goal_worst_sum, prob_goal_best_sum])
                critical    = np.array([prob_critical_worst_sum, prob_critical_best_sum])
                    
                # Compute probability intervals
                probs_lb = np.min(approx, axis=0) - self.setup.main['interval_margin']
                probs_ub = np.max(approx, axis=0) + self.setup.main['interval_margin']
                   
                # Compute deadlock probability intervals
                deadlock_lb = 1-sum(np.max(approx, axis=0)) - max(goal) - max(critical) - self.setup.main['interval_margin']
                deadlock_ub = 1-sum(np.min(approx, axis=0)) - min(goal) - min(critical) + self.setup.main['interval_margin']
                    
                # Compute probability intervals to reach goal
                goal_lb = min(goal) - self.setup.main['interval_margin']
                goal_ub = max(goal) + self.setup.main['interval_margin']    
                
                # Compute approximate probability to reach goal
                goal_approx = np.round(np.mean(goal), nr_decimals)
                    
                critical_lb = min(critical) - self.setup.main['interval_margin']
                critical_ub = max(critical) + self.setup.main['interval_margin']  
                
                # Compute approximate probability to reach critical
                critical_approx = np.round(np.mean(critical), nr_decimals)
                
                # Create interval strings (only entries for prob > 0)
                interval_strings = ["["+
                                  str(floor_decimal(max(1e-4, lb),nr_decimals))+","+
                                  str(floor_decimal(min(1,    ub),nr_decimals))+"]"
                                  for lb,ub in zip(probs_lb, probs_ub)]
                
                deadlock_string = '['+ \
                   str(floor_decimal(max(1e-4, deadlock_lb),nr_decimals))+','+ \
                   str(floor_decimal(min(1,    deadlock_ub),nr_decimals))+']'
                   
                if goal_approx != 0:
                    goal_string = '['+ \
                       str(floor_decimal(max(1e-4, goal_lb),nr_decimals))+','+ \
                       str(floor_decimal(min(1,    goal_ub),nr_decimals))+']'
                
                else:
                    goal_string = None
                    
                if critical_approx != 0:
                    
                    critical_string = '['+ \
                       str(floor_decimal(max(1e-4, critical_lb),nr_decimals))+','+ \
                       str(floor_decimal(min(1,    critical_ub),nr_decimals))+']'
                
                else:
                    critical_string = None
                
                # Create approximate prob. strings (only entries for prob > 0)
                approx_strings = np.mean(approx, axis=0)
                
                # Compute approximate deadlock transition probabilities
                deadlock_approx = np.round(1-sum(approx_strings)-goal_approx-critical_approx, nr_decimals)
                
                prob[a] = {
                    'interval_strings': interval_strings,
                    'interval_idxs': nonzero_idxs,
                    'approx_strings': approx_strings,
                    'approx_idxs': nonzero_idxs,
                    'deadlock_interval_string': deadlock_string,
                    'deadlock_approx': deadlock_approx,
                    'goal_interval_string': goal_string,
                    'goal_approx': goal_approx,
                    'critical_interval_string': critical_string,
                    'critical_approx': critical_approx
                }
                
                # Print normal row in table
                if a % printEvery == 0 and verbose:
                    nr_transitions = len(prob[a]['interval_idxs'])
                    tab.print_row([a, 
                       'Transitions: '+str(nr_transitions)+\
                       ' (skipped: '+str(skipped_counter)+')'])
                
        return prob
    
    def KalmanPrediction(self):
        '''
        Perform the Kalman filter update step (for the progression of the
        covariance matrix)

        Returns
        -------
        None.

        '''
        
        # List to compute best-case covariance
        covsTilde = []
        covsPred  = []
        covs      = []
        
        # Kalman filter list definitions
        self.km[1] = {}#[dict() for i in range(self.N+1)]
        
        # Initial belief of Kalman filter
        self.km[1][0] = {'cov': np.eye(self.system.LTI['n'])*self.system.filter['cov0']}
        
        for k in range(self.N):
            
            # Perform Kalman filter prediction at current time k for delta-type action
            self.km[1][k+1] = kalmanFilter(self.model[self.setup.base_delta], self.km[1][k]['cov'])
        
            covsTilde += [self.km[1][k+1]['cov_tilde']]
            covsPred  += [self.km[1][k+1]['cov_pred']]
            covs      += [self.km[1][k+1]['cov']]
        
        if len(self.setup.jump_deltas) > 0:
        
            self.km['return_step'] = dict()    
        
            # Compute overall best and worst-case covariance matrices
            if self.setup.main['covarianceMode'] == 'SDP':
                func = steadystateCovariance_sdp
            else:
                func = steadystateCovariance    

            # For every possible jump delta value
            for delta_idx, delta in enumerate(self.setup.jump_deltas):    
        
                # Specify the waiting time
                self.km['waiting_time'] = 2
            
                jump_km = [{'cov': [], 'cov_pred': [], 'cov_tilde': [], 'error_bound': []} 
                                  for i in range(self.km['waiting_time'] + 2)]
                
                # For every exact covariance at the base rate
                for i,cov in enumerate(covs):
                    belief = kalmanFilter(self.model[delta], cov)
                    jump_km[0]['cov']         += [belief['cov']]
                    jump_km[0]['cov_pred']    += [belief['cov_pred']]
                    jump_km[0]['cov_tilde']   += [belief['cov_tilde']]
                    jump_km[0]['error_bound'] += [belief['error_bound']]
                
                    # Propagate the waiting time steps (recovering steps)
                    for gamma in range(1, self.km['waiting_time'] + 2):
                        belief = kalmanFilter(self.model[self.setup.base_delta], belief['cov'])
                
                        jump_km[gamma]['cov']         += [belief['cov']]
                        jump_km[gamma]['cov_pred']    += [belief['cov_pred']]
                        jump_km[gamma]['cov_tilde']   += [belief['cov_tilde']]
                        jump_km[gamma]['error_bound'] += [belief['error_bound']]
                 
                self.km[delta] = [dict() for g in range(self.km['waiting_time']+2)]
                for gamma in range(0, self.km['waiting_time'] + 2):
                    
                    # Compute best/worst-case covariances of jump mode
                    jump_limits = func(jump_km[gamma]['cov_pred'], verbose=False)
                    self.km[delta][gamma]['cov_pred_worst'] = jump_limits['worst']
                    self.km[delta][gamma]['cov_pred_best']  = jump_limits['best']
                    
                    jump_limits = func(jump_km[gamma]['cov_tilde'], verbose=False)
                    self.km[delta][gamma]['cov_tilde_worst'] = jump_limits['worst']
                    self.km[delta][gamma]['cov_tilde_best']  = jump_limits['best']
                    
                    self.km[delta][gamma]['error_bound'] = np.max(np.array(jump_km[gamma]['error_bound']), axis=0)

                jump_final_worst = self.km[delta][self.km['waiting_time'] + 1]['cov_pred_worst']
                
                distance = np.zeros(self.N)
                
                from scipy.linalg import eigh
                
                for k in range(self.N):
                    cov = self.km[1][k+1]['cov_pred']
                    
                    lambdas = eigh(cov, jump_final_worst, eigvals_only=True)
                    distance[k] = np.sqrt( sum([np.log(lambd)**2 for lambd in lambdas]) )
                
                self.km['return_step'][delta] = np.minimum( np.argmin(distance),
                                                        self.setup.mdp['k_steady_state'] )
                
                print('After delta',delta,'action, go back to step:',self.km['return_step'][delta])
                        
            # Compute best/worst-case covariances of jump mode
            func([jump_final_worst, covsPred[self.km['return_step'][delta]]], verbose=False)
            
    def defTransitions(self):
        '''
        Define the transition probabilities of the finite-state abstraction 
        (perform for every iteration of the iterative scheme).

        Returns
        -------
        None.

        '''
           
        # Column widths for tabular prints
        col_width = [8,46]
        tab = table(col_width)
        
        self.trans = {'prob': {1: {}}}
                
        # Retreive type of horizon
        if self.setup.mdp['k_steady_state'] == None:
            k_range = np.arange(self.N)
        else:
            k_range = np.arange(self.setup.mdp['k_steady_state'])
        
        print('Computing transition probabilities...')
        
        ######
        
        # Compute transition probabilities at the base rate
        # For every time step in the horizon
        for k in k_range:
            
            k_prime = k + 1
            goal = self.abstr['goal']['X'][k_prime]
            critical = self.abstr['critical']['X'][k_prime]
            
            Sigma_worst = self.km[1][k_prime]['cov_tilde']
            Sigma_best  = self.km[1][k_prime]['cov_tilde']
            
            # Print header row
            if self.setup.main['verbose']:
                tab.print_row(['ACTION','STATUS'], head=True)    
            
            print('Compute transition probabilities for k',k)
            
            self.trans['prob'][1][k_prime] = \
                self._computeProbabilityBounds(tab, Sigma_worst, Sigma_best, self.abstr['actions_inv'][1], goal, critical)
            
        # Delete iterable variables
        del k
        
        # If 2-phase time horizon is enabled...
        if self.setup.mdp['k_steady_state'] != None:
            
            k_prime = self.setup.mdp['k_steady_state'] + 1
            goal = self.abstr['goal']['X'][k_prime]
            critical = self.abstr['critical']['X'][k_prime]
            
            Sigma_worst = self.km[1]['steady']['worst']
            Sigma_best  = self.km[1]['steady']['best']
            
            # Print header row
            if self.setup.main['verbose']:
                tab.print_row(['ACTION','STATUS'], head=True)    
                
            print('Compute transition probabilities for steady-state phase')
                
            # Compute worst/best-case transition probabilities in 
            # steady-state portion of the time horizon
            self.trans['prob'][1][k_prime] = \
                self._computeProbabilityBounds(tab, Sigma_worst, Sigma_best, self.abstr['actions_inv'][1], goal, critical)
                
        ######
        
        # Compute transition probabilities for jump delta values
        
        for delta in self.setup.jump_deltas:
            self.trans['prob'][delta] = dict()
            
            # For every time step in the horizon
            for gamma in range( self.km['waiting_time'] + 1 ):
                
                # Continue with jump delta value
                k_prime = gamma
                goal = self.abstr['goal'][delta][k_prime]
                critical = self.abstr['critical'][delta][k_prime]
                
                Sigma_worst = self.km[delta][k_prime]['cov_tilde_worst']
                Sigma_best  = self.km[delta][k_prime]['cov_tilde_best']
                
                # Print header row
                if self.setup.main['verbose']:
                    tab.print_row(['ACTION','STATUS'], head=True)    
                
                print('Compute transition probabilities for delta',delta)
                
                self.trans['prob'][delta][k_prime] = \
                    self._computeProbabilityBounds(tab, Sigma_worst, Sigma_best, self.abstr['actions_inv'][delta], goal, critical)
                
            # Delete iterable variables
            del gamma
            
        self.time['3_probabilities'] = tocDiff(False)
        print('\nTransition probabilities calculated - time:',
              self.time['3_probabilities'],'\n\n')
        
class MonteCarloSim():
    
    def __init__(self, Ab, iterations=100, init_states=False):
        
        self.results = dict()
        self.traces = dict()
        
        self.setup = Ab.setup
        self.system = Ab.system
        self.km = Ab.km
        self.policy = Ab.mdp.MAIN_DF
        self.abstr = Ab.abstr
        self.horizon = Ab.N
        self.model = Ab.model

        # Column widths for tabular prints
        if self.setup.main['verbose']:
            col_width = [8,6,6,46]
            self.tab = table(col_width)
                                            
        if not init_states:
            self.init_states = np.arange(len(self.abstr['P']))
            
        else:
            self.init_states = init_states
            
        self.results['reachability_probability'] = \
            np.zeros(len(self.init_states))
            
        self.iterations = iterations
        
        #####
        
        self.defineDisturbances()
        self.run()
        
        #####
        
        del self.setup
        del self.system
        del self.km
        del self.policy
        del self.abstr
        del self.horizon
        del self.model
        
    def defineDisturbances(self):
        
        if self.setup.main['verbose']:
            print(' -- Computing required Gaussian random variables...')
        
        # The gaussian random variables are precomputed to speed up the code
        self.noise = dict()
        self.noise['w'] = dict()
        for delta in self.setup.all_deltas:
            self.noise['w'][delta] = np.random.multivariate_normal(
                np.zeros(self.model[delta]['n']), self.model[delta]['noise']['w_cov'],
               ( len(self.init_states), self.iterations, self.horizon ))
    
        self.noise['v'] = np.random.multivariate_normal(
                np.zeros(self.system.LTI['r']), self.system.LTI['noise']['v_cov'], 
               ( len(self.init_states), self.iterations, self.horizon ))
    
    def run(self):
        
        for r_abs, r in enumerate(self.init_states):
            
            if self.setup.main['verbose']:
                self.tab.print_row(['STATE','ITER','K','STATUS'], head=True)
            else:
                print(' -- start Monte Carlo simulations for initial region',r)
            
            # Create dictionaries for results related to partition i
            self.results[r]  = dict()
            self.results[r]['goalReached'] = np.full(self.iterations, False, dtype=bool)
            self.traces[r]   = dict()    
            
            # For each of the monte carlo iterations
            for m in range(self.iterations):
                
                self.traces[r][m], self.results[r]['goalReached'][m] = \
                    self._runIteration(r_abs, r, m)
                    
            self.results['reachability_probability'][r_abs] = \
                np.mean( self.results[r]['goalReached'] )
    
    def _runIteration(self, r_abs, r, m):
        
        success = False
        trace = {'k': [], 'x': [], 'bel_mu': [], 'bel_cov': [], 'y': []}
        
        k = 0
        
        if self.setup.main['verbose']:
            self.tab.print_row([r, m, k, 'Start Monte Carlo iteration'])
        
        # Initialize the current simulation
        x           = np.zeros((self.horizon + 1, self.system.LTI['n']))
        y           = np.zeros((self.horizon + 1, self.system.LTI['r']))
        mu          = np.zeros((self.horizon + 1, self.system.LTI['n']))
        cov         = [None for i in range(self.horizon + 1)]
        mu_goal     = [None for i in range(self.horizon + 1)]
        x_region    = np.zeros(self.horizon + 1).astype(int)
        mu_region   = np.zeros(self.horizon + 1).astype(int)
        u           = [None for i in range(self.horizon)]
        
        actionToTake = np.zeros(self.horizon).astype(int)
        deltaToTake  = np.zeros(self.horizon).astype(int)
        
        # Construct initial belief
        mu[0] = np.array(self.abstr['P'][r]['center'])
        cov[0] = self.km[1][0]['cov']
        
        # Retreive the initial action time-grouping to be chosen
        # (given by the optimal policy to the MDP)
        delta = 1
        
        # Determine initial state and measurement
        w_init = np.random.multivariate_normal(
            np.zeros(self.system.LTI['n']), self.km[delta][0]['cov'])
        v_init = self.noise['v'][r_abs, m, k]
        
        # True state model dynamics
        x[0] = mu[0] + w_init
        y[0] = self.model[delta]['C'] @ np.array(x[0]) + v_init
        
        # Add current state, belief, etc. to trace
        trace['k'] += [0]
        trace['x'] += [x[0]]
        trace['bel_mu'] += [mu[0]]
        trace['bel_cov'] += [self.km[delta][0]['cov']]
        trace['y'] += [y[0]]
        
        ######
        
        abs_state_shift = 0
        
        # For each time step in the finite time horizon
        while k < self.horizon:
            
            # Determine in which region the TRUE state is
            x_cubic = skew2cubic(x[k], self.abstr)
            
            cubic_center_x = computeRegionCenters(x_cubic, 
                                    self.system.partition).flatten()
            
            # Determine in which region the BELIEF MEAN is
            mu_cubic = skew2cubic(mu[k], self.abstr)
            
            cubic_center_mu = computeRegionCenters(mu_cubic, 
                                    self.system.partition).flatten()
            
            # Check if the state is in a goal region
            if self._stateInGoal( point=x_cubic ):
                
                success = True
                if self.setup.main['verbose']:
                    self.tab.print_row([r, m, k, 'Goal state reached'], sort="Success")
                return trace, success
            
            # Check if the state is in a critical region
            if self._stateInCritical( point=x_cubic ):
                
                if self.setup.main['verbose']:
                    self.tab.print_row([r, m, k, 'Critical state reached, so abort'], sort="Warning")
                return trace, success
            
            ###
            
            x_region[k] = self._getStateRegion(cubic_center_x)
            
            if tuple(cubic_center_mu) in self.abstr['allCentersCubic']:
                # Save region of BELIEF MEAN state
                mu_region[k] = self.abstr['allCentersCubic'][tuple(cubic_center_mu)]
                
                # Retreive the action from the policy
                abs_state = abs_state_shift + mu_region[k]
                actionToTake[k] = self.policy['opt_action'][abs_state][k]
                deltaToTake[k]  = self.policy['opt_delta'][abs_state][k]
                
                # Compute the new state shift value
                opt_k_succ_id   = self.policy['opt_ksucc_id'][abs_state][k]
                
                # Here, we automatcially skip trivial waiting actions for delta>1 actions
                abs_state_shift = opt_k_succ_id + (deltaToTake[k]-1)*self.abstr['nr_actions']
                
                if actionToTake[k] == -1:
                    
                    if self.setup.main['verbose']:
                        self.tab.print_row([r, m, k, 'No policy known, so abort'], sort="Warning")
                    return trace, success
                
                elif actionToTake[k] == -2:
                    
                    print('SHOULD NOT HAPPEN!')
                    
                    # If the action value is -2, it's a trivial wait action
                    k += 1
                    continue
                
            else:
                
                if self.setup.main['verbose']:
                    self.tab.print_row([r, m, k, 'Absorbing state reached, so abort'], sort="Warning")
                return trace, success
            
            ###
            
            # If loop was not aborted, we have a valid action
            delta = deltaToTake[k]
            
            if self.setup.main['verbose']:
                
                branch_delta = self.policy['delta'][abs_state]
                
                self.tab.print_row([r, m, k, 'Belief region '+
                                    str(mu_region[k])+', true region '+
                                    str(x_region[k])+' (branch delta='+str(branch_delta)+
                                    '), action ' + str(actionToTake[k])+
                                    ' (delta='+str(delta)+')'])
                
            # Only perform another movement if k < N
            if k < self.horizon - delta:
        
                # Apply 1-step Kalman filter
                belief = kalmanFilter(self.model[delta], cov[k])        
        
                # Move predicted mean to the future belief to the target point of the next state
                mu_goal[k+delta] = self.abstr['target']['d'][actionToTake[k]]
        
                w = self.noise['w'][delta][r_abs, m, k]
                v = self.noise['v'][r_abs, m, k]
        
                # Compute state at the next time step
                x[k+delta], u[k], y[k+delta], mu[k+delta] = \
                    self._computeState(delta, belief['K_gain'],
                                      mu_goal[k+delta], x[k], w, v)
        
                # Update posterior belief covariance
                cov[k+delta] = belief['cov']
        
                if any(self.model[delta]['uMin'] > u[k]) or \
                   any(self.model[delta]['uMax'] < u[k]):
                    self.tab.print_row([r, m, k, 'Control input '+str(u[k])+' outside limits'], sort="Warning")
        
                # Add current state, belief, etc. to trace
                trace['k'] += [k+delta]
                trace['x'] += [x[k+delta]]
                trace['bel_mu'] += [mu[k+delta]]
                trace['bel_cov'] += [belief['cov']]
                trace['y'] += [y[k+delta]]
            
            # Increase iterator variable by the value of delta associated to chosen action
            k += delta
                        
        ######
        
        return trace, success
    
    def _stateInGoal(self, point):
        
        if any([in_hull( point, block['hull'] ) for block in self.system.spec['goal'].values()]):
            return True
        else:
            return False
        
    def _stateInCritical(self, point):
    
        if any([in_hull( point, block['hull'] ) for block in self.system.spec['critical'].values()]):
            return True
        else:
            return False
    
    def _getStateRegion(self, x):
        
        if tuple(x) in self.abstr['allCentersCubic']:
            # Save region of TRUE state 
            return self.abstr['allCentersCubic'][tuple(x)]
            
        else:
            return -1
    
    def _computeState(self, delta, K_gain, mu_goal, x, w, v):
        
        # Reconstruct the control input required to achieve this target point
        # Note that we do not constrain the control input; we already know that a suitable control exists!
        u = np.array(self.model[delta]['B_pinv'] @ ( mu_goal - self.model[delta]['A'] @ x - self.model[delta]['Q_flat'] ))
        
        # Implement the control into the physical (unobservable) system
        x_hat = self.model[delta]['A'] @ x + self.model[delta]['B'] @ u + self.model[delta]['Q_flat']
        
        # Use Gaussian process noise
        x_prime = x_hat + w
            
        # Generate a measurement of the true state
        y_prime = self.model[delta]['C'] @ x_prime + v
        
        # Update the belief based on action and measurement                    
        mu_prime = mu_goal + K_gain @ (y_prime - self.model[delta]['C'] @ mu_goal)
        
        return x_prime, u, y_prime, mu_prime
    
    
    
    
    