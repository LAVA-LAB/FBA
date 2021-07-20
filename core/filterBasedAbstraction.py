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
    floor_decimal, extractRowBlockDiag

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
            # self.abstr['P'][a]['euclidDist'] = np.hypot(*(self.abstr['target']['d'][a] - allCentersArray).T)
            # self.abstr['P'][a]['priority'] = [i for i,d in enumerate(self.abstr['P'][a]['distances']) if sum(d == 0) == len(d)-1]
        
    def _computeProbabilityBounds(self, tab, k, delta, verbose=True):
        '''
        Compute transition probability intervals (bounds)

        Parameters
        ----------
        tab : dict
            Table dictionary.
        k : int
            Discrete time step.
        delta : int
            Value of delta to use the model for.

        Returns
        -------
        prob : dict
            Dictionary containing the computed transition probabilities.

        '''

        printEvery = min(100, max(1, int(self.abstr['nr_actions']/10)))

        nr_decimals = 5 # Number of decimals to round off on
        threshold_decimals = 4 # Minimum probability to model

        # Determine minimum action time step width
        min_delta = min(self.setup.deltas)
        
        if self.setup.lic['enabled'] and delta > min_delta and k not in ['steadystate']:

            print(' -- Copy transition probabilities for',delta,'at time',k)
            
            # Copy transition probabilities
            prob = self.trans['prob'][min_delta][k]
                    
        else:
            
            # Determine covariance of distribution                
            if k == 'steadystate':
                Sigma_worst = self.km[delta]['steady']['worst']
                Sigma_best  = self.km[delta]['steady']['best']
                
                SigmaEqual = False
                
            elif self.setup.lic['enabled']:
                # Retreive waiting time
                gamma = self.km[delta]['F'][k]['waiting_time']
                
                Sigma_worst = self.km[delta]['F'][k+delta+gamma]['cov_tilde']
                Sigma_best  = self.km['cov_tilde_min']
                
                SigmaEqual = False
                
            else:
                Sigma_worst = self.km[delta]['F'][k+delta]['cov_tilde']
                Sigma_best  = self.km[delta]['F'][k+delta]['cov_tilde']
                
                SigmaEqual = True
                
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
                if len(self.abstr['actions_inv'][delta][a]) > 0:
                        
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
                        
                        if k == 'steadystate':
                            k_prime = self.setup.mdp['k_steady_state'] + delta
                        else:
                            k_prime = k + delta
                        
                        ### 2) Probability to reach goal state
                        # Subtract the probability to end up in a goal state
                        if j in self.abstr['goal'][k_prime] and (probs_worst > 0 or probs_best > 0):
                            
                            all_limits = self.abstr['goal'][k_prime][j]
                        
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
                        if j in self.abstr['critical'][k_prime] and (probs_worst > 0 or probs_best > 0):
                            
                            all_limits = self.abstr['critical'][k_prime][j]
                                
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
                        tab.print_row([delta, k, a, 
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
        
        # Column widths for tabular prints
        if self.setup.main['verbose']:
            col_width = [8,8,46]
            tab = table(col_width)
            
        max_error_bound = np.zeros((len(self.setup.deltas), self.N+max(self.setup.deltas), self.system.LTI['n']))
        
        if self.setup.lic['enabled']:
            # Determine maximum threshold for measure on covariance over time
            maxA = self.setup.lic['LICMaxA']
            minA = self.setup.lic['LICMinA']
            self.km['maxEllipseSize'] = [i*minA + (1-i)*maxA for i in np.arange(0,1,1/self.N)]
            
        # List to compute best-case covariance
        covTildeAll = []
            
        for delta_idx,delta in enumerate(self.setup.deltas):
            
            # Print header row
            if self.setup.main['verbose']:
                tab.print_row(['DELTA','K','STATUS'], head=True)
            
            # Kalman filter list definitions
            self.km[delta] = dict()
            self.km[delta]['F'] = [dict() for i in range(self.N+delta)]
            
            # Initial belief of Kalman filter
            for i in range(delta):
                self.km[delta]['F'][i]['cov'] = np.eye(self.system.LTI['n'])*self.system.filter['cov0']
            
            # For every time step in the finite horizon
            for k in range(self.N):
                
                # Perform Kalman filter prediction at current time k for delta-type action
                self.km[delta]['F'][k+delta], max_error_bound[delta_idx,k+delta, :] = \
                    kalmanFilter(self.model[delta], self.km[delta]['F'][k]['cov'])
                
                # Set waiting time to zeor by default
                self.km[delta]['F'][k]['waiting_time']  = 0
                
                # If local information controller is active
                if self.setup.lic['enabled']:
                    
                    covTildeAll += [np.copy(self.km[delta]['F'][k+delta]['cov_tilde'])]
                    
                    # Then, a best/worst-case covariance matrix is used, and
                    # we force the actual covariance always to below this
                    
                    # Set the worst-case covariance matrix to the circle with  
                    # radius equal to the maximum measure (i.e. threshold)
                    self.km[delta]['F'][k+delta]['cov_tilde'] = \
                        np.eye(self.system.LTI['n']) * self.km['maxEllipseSize'][k]**2
                    
                    # If the actual measure if above the threshold, perform a
                    # "wait action" (if possible) to reduce it
                    if self.km[delta]['F'][k+delta]['cov_tilde_measure'] > self.km['maxEllipseSize'][k]:
                        # Reduce the uncertainty after taking the action, until it's below the threshold
                        
                        # Set waiting time equal to the worst-case possible
                        self.km[delta]['F'][k+delta]['waiting_time'] = self.N
                        
                        # Determine how much stabilization time is needed to steer below the threshold
                        gamma = 0
                        delta_lowest = min(self.setup.deltas)
                                                
                        # Copy the measure on the covariance matrix
                        covStabilize = self.km[delta]['F'][k+delta]['cov']
                        covMeasure = self.km[delta]['F'][k+delta]['cov_tilde_measure']
                        
                        flag = False
                        
                        while not flag and k + delta + gamma*delta_lowest < self.N:
                            if covMeasure <= self.km['maxEllipseSize'][k+gamma*delta_lowest]:
                                # If measure is now below threshold, we're okay
                                flag = True
                                if self.setup.main['verbose']:
                                    print('Have to wait',gamma,'steps of width',delta_lowest)
                                    print('Measure is reduced to',covMeasure)
                                
                                # Store how long we must wait to reduce uncertainty
                                self.km[delta]['F'][k]['waiting_time'] = np.copy(gamma)
                                
                            else:
                                if self.setup.main['verbose']:
                                    print('Covariance measure too high at',covMeasure,'; gamma is',gamma)
                                
                                # Increment gamma value (stabilization time)
                                gamma += 1
                                
                                # Determine resulting covariance after wait
                                #_, _, _, covStabilize, covMeasure, _ = \
                                out, _ = \
                                    kalmanFilter(self.model[delta_lowest], covStabilize)                            
                                
                                covStabilize = out['cov']
                                covMeasure   = out['cov_tilde_measure']
                    
                # Print normal row in table
                if self.setup.main['verbose']:
                    tab.print_row([delta, k, 'Measure: '+
                       str(np.round(self.km[delta]['F'][k+delta]['cov_tilde_measure'], decimals=3))])
                    tab.print_row([delta, k, 'Error bound (epsilon): '+
                       str(np.round(max_error_bound[delta_idx, k+delta], decimals=3))])
                
            # Delete iterable variable
            del k
        del delta
        
        # Compute error bound by which obstacles are augmented
        self.km['max_error_bound']   = np.max(max_error_bound, axis=0)
        
        # Compute best-case covariance (for local information controller)
        if self.setup.lic['enabled']:
            if self.setup.main['covarianceMode'] == 'SDP':
                func = steadystateCovariance_sdp
            else:
                func = steadystateCovariance
            
            self.km['cov_tilde_min'] = func(covTildeAll, verbose=False)['best']
    
    def defTransitions(self):
        '''
        Define the transition probabilities of the finite-state abstraction 
        (perform for every iteration of the iterative scheme).

        Returns
        -------
        None.

        '''
           
        # Column widths for tabular prints
        col_width = [8,8,8,46]
        tab = table(col_width)
        
        self.trans = {'prob': {}}
                
        # Retreive type of horizon
        if self.setup.mdp['k_steady_state'] == None:
            k_range = np.arange(self.N)
        else:
            k_range = np.arange(self.setup.mdp['k_steady_state'])
        
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
                    print(' -- Calculate transitions for delta',
                          delta,'at time',k)
                
                self.trans['prob'][delta][k] = \
                    self._computeProbabilityBounds(tab, k, delta)
                
            # Delete iterable variables
            del k
            
            # If 2-phase time horizon is enabled...
            if self.setup.mdp['k_steady_state'] != None:
                
                # Print header row
                if self.setup.main['verbose']:
                    tab.print_row(['DELTA','K','ACTION','STATUS'], head=True)    
                    
                # Compute worst/best-case transition probabilities in 
                # steady-state portion of the time horizon
                self.trans['prob'][delta][self.setup.mdp['k_steady_state']] = \
                    self._computeProbabilityBounds(tab, 'steadystate', delta)
            
        del delta
        
        self.time['3_probabilities'] = tocDiff(False)
        print('Transition probabilities calculated - time:',
              self.time['3_probabilities'])

    def monteCarlo(self, iterations='auto', init_states='auto', 
                   init_times='auto'):
        '''
        Perform Monte Carlo simulations to validate the obtained results

        Parameters
        ----------
        iterations : str or int, optional
            Number of Monte Carlo iterations. The default is 'auto'.
        init_states : str or list, optional
            Initial states to start simulations from. The default is 'auto'.
        init_times : str or list, optional
            Initial time steps to start sims. from. The default is 'auto'.

        Returns
        -------
        None.

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
            
        n_list = self.setup.montecarlo['init_timesteps']
        
        self.mc['results']['reachability_probability'] = \
            np.zeros((len(self.abstr['P']),len(n_list)))
        
        # Column widths for tabular prints
        if self.setup.main['verbose']:
            col_width = [6,8,6,6,46]
            tab = table(col_width)

            print(' -- Computing required Gaussian random variables...')
        
        if self.setup.montecarlo['init_states'] == False:
            init_state_idxs = np.arange(len(self.abstr['P']))
            
        else:
            init_state_idxs = self.setup.montecarlo['init_states']
        
        # The gaussian random variables are precomputed to speed up the code
        if self.setup.scenarios['gaussian'] is True:
            w_array = dict()
            for delta in self.setup.deltas:
                w_array[delta] = np.random.multivariate_normal(
                    np.zeros(self.model[delta]['n']), self.model[delta]['noise']['w_cov'],
                   ( len(n_list), len(init_state_idxs), self.setup.montecarlo['iterations'], self.N ))
        
            v_array = np.random.multivariate_normal(
                    np.zeros(self.system.LTI['r']), self.system.LTI['noise']['v_cov'], 
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
                    
                    self.mc['traces'][n0][i][m] = {'k': [], 'x': [], 'bel_mu': [], 'bel_cov': [], 'y': []}
                    
                    # Retreive the initial action time-grouping to be chosen
                    # (given by the optimal policy to the MDP)
                    delta = self.results['optimal_delta'][n0,i]
                    
                    if False :#i in self.abstr['goal'][n0]:
                        # If the initial state is already the goal state, succes
                        # Then abort the current iteration, as we have achieved the goal
                        self.mc['results'][n0][i]['goalReached'][m] = True
                        
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 'Initial state is goal state'], sort="Success")
                    elif delta == -1:
                        # If delta is zero, no policy is known, and reachability is zero
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 'No initial policy known, so abort'], sort="Warning")
                    else:
                        if self.setup.main['verbose']:
                            tab.print_row([n0, i, m, n0, 'Start Monte Carlo iteration'])
                                            
                        # Initialize the current simulation
                        
                        x = np.zeros((self.N+1, self.system.LTI['n']))
                        y = np.zeros((self.N, self.system.LTI['r']))
                        mu = np.zeros((self.N, self.system.LTI['n']))
                        mu_goal = [None for i in range(self.N+1)]
                        x_region = np.zeros(self.N+1).astype(int)
                        mu_region = np.zeros(self.N+1).astype(int)
                        u = [None for i in range(self.N)]
                        
                        actionToTake = np.zeros(self.N).astype(int)
                        deltaToTake = np.zeros(self.N).astype(int)
                        
                        # Set initial time
                        k = n0
                        
                        # Construct initial belief
                        mu[n0] = np.array(regionA['center'])
                        
                        # Determine initial state and measurement
                        w_init = np.random.multivariate_normal(
                            np.zeros(self.system.LTI['n']), self.km[delta]['F'][n0]['cov'])
                        v_init = v_array[n0id, i_abs, m, k]
                        
                        # True state model dynamics
                        x[n0] = mu[n0] + w_init
                        y[n0] = self.model[delta]['C'] @ np.array(x[n0]) + v_init
                        
                        # Add current state, belief, etc. to trace
                        self.mc['traces'][n0][i][m]['k'] += [n0]
                        self.mc['traces'][n0][i][m]['x'] += [x[n0]]
                        self.mc['traces'][n0][i][m]['bel_mu'] += [mu[n0]]
                        self.mc['traces'][n0][i][m]['bel_cov'] += [self.km[delta]['F'][n0]['cov']]
                        self.mc['traces'][n0][i][m]['y'] += [y[n0]]
                        
                        # For each time step in the finite time horizon
                        while k < self.N / min(self.setup.deltas):
                            
                            if self.setup.main['verbose']:
                                tab.print_row([n0, i, m, k, 'New time step'])
                            
                            # Determine in which region the TRUE state is
                            x_cubic = skew2cubic(x[k], self.abstr)
                            
                            cubic_center_x = computeRegionCenters(x_cubic, 
                                                    self.system.partition).flatten()
                            
                            # Determine in which region the BELIEF MEAN is
                            mu_cubic = skew2cubic(mu[k], self.abstr)
                            
                            cubic_center_mu = computeRegionCenters(mu_cubic, 
                                                    self.system.partition).flatten()
                            
                            ###
                            
                            # Check if the state is in a goal region
                            if any([in_hull( x_cubic, block['hull'] ) for block in self.system.spec['goal'].values()]):
                                
                                # Then abort the current iteration, as we have achieved the goal
                                self.mc['results'][n0][i]['goalReached'][m] = True
                                
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Goal state reached'], sort="Success")
                                break  
                            
                            # Check if the state is in a critical region
                            if any([in_hull( x_cubic, block['hull'] ) for block in self.system.spec['critical'].values()]):
                                
                                # Then abort current iteration
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Critical state reached, so abort'], sort="Warning")
                                break
                            
                            ###
                            
                            if tuple(cubic_center_x) in self.abstr['allCentersCubic']:
                                # Save region of TRUE state
                                x_region[k] = self.abstr['allCentersCubic'][tuple(cubic_center_x)]
                                
                            else:
                                x_region[k] = -1
                                
                            if tuple(cubic_center_mu) in self.abstr['allCentersCubic']:
                                # Save region of BELIEF MEAN state
                                mu_region[k] = self.abstr['allCentersCubic'][tuple(cubic_center_mu)]
                                
                                # Retreive the action from the policy
                                actionToTake[k] = self.results['optimal_policy'][k, mu_region[k]]
                                
                            else:
                                mu_region[k] = -1
                            
                            if mu_region[k] == -1:
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
                            deltaToTake[k] = self.results['optimal_delta'][k, mu_region[k]]
                            delta = deltaToTake[k]
                            
                            # If delta is zero, no policy is known, and reachability is zero
                            if delta == 0:
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Action type undefined, so abort'], sort="Warning")
                                break
                            
                            if self.setup.lic['enabled']:
                                # If the local information controller is active,
                                # also account for the "waiting time"
                                
                                # Retreive gamma value
                                gamma = self.km[delta]['F'][k]['waiting_time']
                                
                                # Multiply with the minimum time step width to get waiting time
                                waiting_time = gamma*min_delta
                                
                                # Print message
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Wait for '+str(gamma)+'*'+\
                                               str(min_delta)+'='+str(waiting_time)+' time steps'])
                            else:
                                # If local information controller is not active,
                                # simply set the waiting time to zero
                                gamma = 0
                                waiting_time = 0
                                
                            if self.setup.main['verbose']:
                                tab.print_row([n0, i, m, k, 'Belief state '+str(mu_region[k])+', true state '+str(x_region[k])+', action '+str(actionToTake[k])+' (delta='+str(delta)+')'])
                            
                            # Only perform another movement if k < N
                            if k < self.N - delta:
                            
                                # Move predicted mean to the future belief to the target point of the next state
                                mu_goal[k+delta] = self.abstr['target']['d'][actionToTake[k]]
        
                                # Reconstruct the control input required to achieve this target point
                                # Note that we do not constrain the control input; we already know that a suitable control exists!
                                u[k] = np.array(self.model[delta]['B_pinv'] @ ( mu_goal[k+delta] - self.model[delta]['A'] @ x[k] - self.model[delta]['Q_flat'] ))
                                
                                if any(self.model[delta]['uMin'] > u[k]) or \
                                   any(self.model[delta]['uMax'] < u[k]):
                                    tab.print_row([n0, i, m, k, 'Control input '+str(u[k])+' outside limits'], sort="Warning")
                                
                                # Implement the control into the physical (unobservable) system
                                x_hat = self.model[delta]['A'] @ x[k] + self.model[delta]['B'] @ u[k] + self.model[delta]['Q_flat']
                                
                                if self.setup.scenarios['gaussian'] is True:
                                    # Use Gaussian process noise
                                    x[k+delta] = x_hat + w_array[delta][n0id, i_abs, m, k]
                                else:
                                    # Use generated samples                                    
                                    disturbance = random.choice(self.model[delta]['noise']['samples'])
                                    
                                    x[k+delta] = x_hat + disturbance
                                    
                                # Generate a measurement of the true state
                                y[k+delta] = self.model[delta]['C'] @ x[k+delta] + v_array[n0id, i_abs, m, k]
                                
                                # Update the belief based on action and measurement                    
                                mu[k+delta] = mu_goal[k+delta] + self.km[delta]['F'][k+delta]['K_gain'] @ \
                                                        (y[k+delta] - self.model[delta]['C'] @ mu_goal[k+delta])    
                                
                                # Add current state, belief, etc. to trace
                                self.mc['traces'][n0][i][m]['k'] += [k+delta]
                                self.mc['traces'][n0][i][m]['x'] += [x[k+delta]]
                                self.mc['traces'][n0][i][m]['bel_mu'] += [mu[k+delta]]
                                self.mc['traces'][n0][i][m]['bel_cov'] += [self.km[delta]['F'][k+delta]['cov']]
                                self.mc['traces'][n0][i][m]['y'] += [y[k+delta]]
                            
                                # Update the belief for the waiting time
                                for g in range(gamma):
                                    # Currently at time k, plus delta of action, plus waiting time
                                    kg = k + delta + (g+1)*min_delta
                                    
                                    # If waiting does not fit in time horizon anymore, break
                                    if kg >= self.N:
                                        if self.setup.main['verbose']:
                                            tab.print_row([n0, i, m, kg-min_delta, 'Time horizon exceeded by waiting, so abort'], sort="Warning")
                                        break
                                    
                                    if self.setup.main['verbose']:
                                        tab.print_row([n0, i, m, kg-min_delta, 'Wait until time: '+str(kg)])
                                        
                                    # Reconstruct the control input required to achieve this target point
                                    # Note that we do not constrain the control input; we already know that a suitable control exists!
                                    u[kg] = np.array(self.model[min_delta]['B_pinv'] @ ( mu_goal[k+delta] - self.model[min_delta]['A'] @ mu[kg-min_delta] - self.model[min_delta]['Q_flat'] ))
                                    
                                    if any(self.model[min_delta]['uMin'] > u[kg]) or \
                                       any(self.model[min_delta]['uMax'] < u[kg]):
                                        tab.print_row([n0, i, m, kg-min_delta, 'Control input '+str(u[kg])+' outside limits'], sort="Warning")
                                    
                                    # Implement the control into the physical (unobservable) system
                                    x_hat = self.model[min_delta]['A'] @ x[kg-min_delta] + self.model[min_delta]['B'] @ u[kg] + self.model[min_delta]['Q_flat']
                                    
                                    if self.setup.scenarios['gaussian'] is True:
                                        # Use Gaussian process noise
                                        x[kg] = x_hat + w_array[min_delta][n0id, i_abs, m, kg]
                                    else:
                                        # Use generated samples                                    
                                        disturbance = random.choice(self.model[min_delta]['noise']['samples'])
                                        
                                        x[kg] = x_hat + disturbance
                                    
                                    # Since we wait in the same state, dynamics are changed accordingly
                                    y[kg] = self.model[min_delta]['C'] @ x[kg] + v_array[n0id, i_abs, m, kg]
                                    
                                    # Update the belief based on action and measurement
                                    mu[kg] = mu_goal[k+delta] + self.km[delta]['F'][kg]['K_gain'] @ \
                                                            (y[kg] - self.model[min_delta]['C'] @ mu_goal[k+delta])
                            
                            # Increase iterator variable by the value of delta associated to chosen action
                            k += delta + waiting_time
                        
                # Set of monte carlo iterations completed for specific initial state
                
                # Calculate the overall reachability probability for initial state i
                self.mc['results']['reachability_probability'][i,n0id] = \
                    np.sum(self.mc['results'][n0][i]['goalReached']) / self.setup.montecarlo['iterations']
                    
        self.time['6_MonteCarlo'] = tocDiff(False)
        print('Monte Carlo simulations finished:',self.time['6_MonteCarlo'])