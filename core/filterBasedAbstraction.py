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
import csv                      # Import to create/load CSV files
import sys                      # Allows to terminate the code at some point
import os                       # Import OS to allow creationg of folders
import random                   # Import to use random variables
from scipy.stats import mvn

from .mainFunctions import computeScenarioBounds_sparse, defineDistances, \
    computeRegionCenters, cubic2skew, skew2cubic, kalmanFilter
from .commons import tic, ticDiff, tocDiff, table, printWarning, \
    floor_decimal, extractRowBlockDiag

from .createMDP import mdp

from .abstraction import Abstraction

class filterBasedAbstraction(Abstraction):
    def __init__(self, setup, basemodel):
        '''
        Initialize filter-based abstraction (FiAb) object

        Parameters
        ----------
        setup : dict
            Setup dictionary.
        basemodel : dict
            Base model for which to create the abstraction.

        Returns
        -------
        None.

        '''
        
        # Copy setup to internal variable
        self.setup = setup
        self.basemodel = basemodel
        
        # Define empty dictionary for monte carlo sims. (even if not used)
        self.mc = dict()   
        
        # Start timer
        tic()
        ticDiff()
        self.time = dict()
        
        Abstraction.__init__(self)
        
        Abstraction.definePartition(self)
        
        # Compute array of all region centers
        allCentersArray = np.array([region['center'] for region in self.abstr['P'].values()])
        
        for a in range(self.abstr['nr_actions']):
            
            self.abstr['P'][a]['distances'] = self.abstr['target']['d'][a] - allCentersArray
            
            # self.abstr['P'][a]['distances'] = defineDistances(self.abstr['P'], 
            #                                   self.abstr['target']['d'][a])
        
    def _computeProbabilityBounds(self, tab, k, delta):
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

        nr_decimals = 5

        # Determine minimum action time step width
        min_delta = min(self.basemodel.setup['deltas'])
            
        if self.setup.lic['enabled'] and delta > min_delta and k not in ['steadystate']:

            print(' -- Copy transition probabilities for',delta,'at time',k)
            
            # Copy transition probabilities
            prob = self.abstr['prob'][min_delta][k]
                    
        else:
            
            # (Re)initialize probability memory dictionary
            prob = dict()
            if k == 'steadystate':
                prob_worst_mem = dict()
                prob_best_mem  = dict()
            else:
                prob_mem = dict()
            
            # For every action (i.e. target point)
            for a in range(self.abstr['nr_actions']):
            
                # Check if action a is available in any state at all
                if len(self.abstr['actions_inv'][delta][a]) > 0 or \
                    self.setup.lic['enabled']:
                        
                    prob[a] = dict()

                    # Determine mean and covariance of distribution                
                    mu = self.abstr['target']['d'][a]
                    if k == 'steadystate':
                        Sigma_worst = self.km[delta]['steady']['worst']
                        Sigma_best = self.km[delta]['steady']['best']
                        
                        # Transform samples back to hypercubic partition
                        muCubic = skew2cubic(mu, self.abstr)
                        SigmaCubic_worst = self.abstr['basis_vectors_inv'].T @ Sigma_worst @ self.abstr['basis_vectors_inv'] 
                        SigmaCubic_best = self.abstr['basis_vectors_inv'].T @ Sigma_best @ self.abstr['basis_vectors_inv'] 
                        
                        probs_worst_all = np.zeros(self.abstr['nr_regions'])
                        probs_best_all  = np.zeros(self.abstr['nr_regions'])
                        
                        # For every possible resulting region
                        for j in range(self.abstr['nr_regions']):
                        
                            # Compute the vector difference between the target point
                            # and the current region
                            coord_distance = tuple(self.abstr['P'][a]['distances'][j])
                            coord_distance_min = tuple(-self.abstr['P'][a]['distances'][j])    
                        
                            # Check if we already calculated this one
                            if coord_distance in prob_worst_mem and not self.setup.main['skewed']:
                                
                                # If so, copy from previous
                                probs_worst_all[j] = prob_worst_mem[coord_distance]
                                probs_best_all[j]  = prob_best_mem[coord_distance]
                                
                            elif coord_distance_min in prob_worst_mem and not self.setup.main['skewed']:
                                
                                # If so, copy from previous
                                probs_worst_all[j] = prob_worst_mem[coord_distance_min]
                                probs_best_all[j]  = prob_best_mem[coord_distance_min]
                                
                            else:    
                        
                                probs_worst_all[j] = floor_decimal(mvn.mvnun(
                                    self.abstr['P'][j]['low'], self.abstr['P'][j]['upp'],
                                    muCubic, SigmaCubic_worst)[0], 4)
                                
                                probs_best_all[j] = floor_decimal(mvn.mvnun(
                                    self.abstr['P'][j]['low'], self.abstr['P'][j]['upp'],
                                    muCubic, SigmaCubic_best)[0], 4)
                                
                                if not self.setup.main['skewed']:
                                    prob_worst_mem[coord_distance] = probs_worst_all[j]
                                    prob_best_mem[coord_distance]  = probs_best_all[j]
                        
                        nonzero_idxs = np.array([idx for idx,(ub1,ub2) in enumerate(zip(probs_worst_all, probs_best_all)) 
                                                 if ub1 > 0 or ub2 > 0])
                        
                        probs_lb = np.minimum(probs_worst_all[nonzero_idxs], probs_best_all[nonzero_idxs]) \
                                    - self.setup.belief['interval_margin']
                        probs_ub = np.maximum(probs_worst_all[nonzero_idxs], probs_best_all[nonzero_idxs]) \
                                    + self.setup.belief['interval_margin']
                        
                        # Compute deadlock probability intervals
                        deadlock_lb = 1-sum( np.maximum(probs_worst_all[nonzero_idxs], probs_best_all[nonzero_idxs]) ) \
                                    - self.setup.belief['interval_margin']
                        deadlock_ub = 1-sum( np.minimum(probs_worst_all[nonzero_idxs], probs_best_all[nonzero_idxs]) ) \
                                    + self.setup.belief['interval_margin']
                                    
                        probs_approx_all = np.mean((probs_worst_all, probs_best_all), axis=0)
                        
                    else:
                        Sigma = self.km[delta]['cov_tilde'][k+delta]
                        
                        # Transform samples back to hypercubic partition
                        muCubic = skew2cubic(mu, self.abstr)
                        SigmaCubic = self.abstr['basis_vectors_inv'].T @ Sigma @ self.abstr['basis_vectors_inv'] 
                    
                        probs_approx_all = np.zeros(self.abstr['nr_regions'])
                        
                        # For every possible resulting region
                        for j in range(self.abstr['nr_regions']):
                        
                            # Compute the vector difference between the target point
                            # and the current region
                            coord_distance = tuple(self.abstr['P'][a]['distances'][j])
                            coord_distance_min = tuple(-self.abstr['P'][a]['distances'][j])    
                        
                            # Check if we already calculated this one
                            if coord_distance in prob_mem and not self.setup.main['skewed']:
                                
                                # If so, copy from previous
                                probs_approx_all[j] = prob_mem[coord_distance]
                                
                            elif coord_distance_min in prob_mem and not self.setup.main['skewed']:
                                
                                # If so, copy from previous
                                probs_approx_all[j] = prob_mem[coord_distance_min]
                                
                            else:
                        
                                probs_approx_all[j] = floor_decimal(mvn.mvnun(
                                    self.abstr['P'][j]['low'], self.abstr['P'][j]['upp'],
                                    muCubic, SigmaCubic)[0], 4)
                                
                                if not self.setup.main['skewed']:
                                    prob_mem[coord_distance] = probs_approx_all[j]
                        
                        nonzero_idxs = np.array([idx for idx,ub in enumerate(probs_approx_all) if ub > 0])
                        
                        probs_lb = probs_approx_all[nonzero_idxs] - self.setup.belief['interval_margin']
                        probs_ub = probs_approx_all[nonzero_idxs] + self.setup.belief['interval_margin']
                    
                        # Compute deadlock probability intervals
                        deadlock_lb = 1-sum(probs_approx_all[nonzero_idxs]) - self.setup.belief['interval_margin']
                        deadlock_ub = 1-sum(probs_approx_all[nonzero_idxs]) + self.setup.belief['interval_margin']
                    
                    # Create interval strings (only entries for prob > 0)
                    interval_strings = ["["+
                                      str(floor_decimal(max(1e-4, lb),nr_decimals))+","+
                                      str(floor_decimal(min(1,    ub),nr_decimals))+"]"
                                      for lb,ub in zip(probs_lb, probs_ub)]
                    
                    deadlock_string = '['+ \
                       str(floor_decimal(max(1e-4, deadlock_lb),nr_decimals))+','+ \
                       str(floor_decimal(min(1,    deadlock_ub),nr_decimals))+']'
                    
                    # Create approximate prob. strings (only entries for prob > 0)
                    approx_strings = [str(p) for p in probs_approx_all[nonzero_idxs]]
                    
                    # Compute approximate deadlock transition probabilities
                    deadlock_approx = np.round(1-sum(probs_approx_all[nonzero_idxs]), nr_decimals)
                    
                    prob[a] = {
                        'interval_strings': interval_strings,
                        'interval_idxs': nonzero_idxs,
                        'approx_strings': approx_strings,
                        'approx_idxs': nonzero_idxs,
                        'deadlock_interval_string': deadlock_string,
                        'deadlock_approx': deadlock_approx,
                    }
                    
                    # Print normal row in table
                    if a % printEvery == 0:
                        nr_transitions = len(prob[a]['interval_idxs'])
                        tab.print_row([delta, k, a, 
                           'Probabilities computed (transitions: '+
                           str(nr_transitions)+')'])
                
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
        
        self.km = dict()
        
        if self.setup.lic['enabled']:
            # Determine maximum threshold for measure on covariance over time
            maxA = self.setup.lic['LICMaxA']
            minA = self.setup.lic['LICMinA']
            self.km['maxEllipseSize'] = [i*minA + (1-i)*maxA for i in np.arange(0,1,1/self.N)]
            
        for delta in self.setup.deltas:
            
            # Print header row
            if self.setup.main['verbose']:
                tab.print_row(['DELTA','K','STATUS'], head=True)
            
            # Kalman filter list definitions
            self.km[delta]                      = dict()
            self.km[delta]['waiting_time']      = dict()
            self.km[delta]['cov_pred']          = [None] * (self.N+delta)
            self.km[delta]['cov']               = [None] * (self.N+delta)
            self.km[delta]['cov_tilde']         = [None] * (self.N+delta)
            self.km[delta]['cov_tilde_measure'] = [None] * (self.N+delta)
            self.km[delta]['max_error_bound']   = [None] * (self.N+delta)
            self.km[delta]['K_gain']            = [None] * (self.N+delta)
            
            # Initial belief of Kalman filter
            for i in range(delta):
                self.km[delta]['cov'][i] = np.eye(self.basemodel.n)*self.setup.belief['cov0']
            
            # For every time step in the finite horizon
            for k in range(self.N):
                
                # Set waiting time to zeor by default
                self.km[delta]['waiting_time'][k]  = 0
                
                # Perform Kalman filter prediction at current time k for delta-type action
                self.km[delta]['cov_pred'][k+delta], self.km[delta]['K_gain'][k+delta], \
                self.km[delta]['cov_tilde'][k+delta], self.km[delta]['cov'][k+delta], \
                self.km[delta]['cov_tilde_measure'][k+delta], self.km[delta]['max_error_bound'][k+delta] = \
                    kalmanFilter(self.model[delta], self.km[delta]['cov'][k])
                
                # If local information controller is active
                if self.setup.lic['enabled']:
                    # Then, a "worst-case" covariance matrix is used, and we
                    # force the actual covariance always to below this
                    
                    # Set the worst-case covariance matrix to the circle with  
                    # radius equal to the maximum measure (i.e. threshold)
                    self.km[delta]['cov_tilde'][k+delta] = \
                        np.eye(self.basemodel.n) * self.km['maxEllipseSize'][k]**2
                    
                    # If the actual measure if above the threshold, perform a
                    # "wait action" (if possible) to reduce it
                    if self.km[delta]['cov_tilde_measure'][k+delta] > self.km['maxEllipseSize'][k]:
                        # Reduce the uncertainty after taking the action, until it's below the threshold
                        
                        # Set waiting time equal to the worst-case possible
                        self.km[delta]['waiting_time'][k] = self.N
                        
                        # Determine how much stabilization time is needed to steer below the threshold
                        gamma = 0
                        delta_lowest = min(self.setup.deltas)
                                                
                        # Copy the measure on the covariance matrix
                        covStabilize = self.km[delta]['cov'][k+delta]
                        covMeasure = self.km[delta]['cov_tilde_measure'][k+delta]
                        
                        flag = False
                        
                        while not flag and k + delta + gamma*delta_lowest < self.N:
                            if covMeasure <= self.km['maxEllipseSize'][k+gamma*delta_lowest]:
                                # If measure is now below threshold, we're okay
                                flag = True
                                if self.setup.main['verbose']:
                                    print('Have to wait',gamma,'steps of width',delta_lowest)
                                    print('Measure is reduced to',covMeasure)
                                
                                # Store how long we must wait to reduce uncertainty
                                self.km[delta]['waiting_time'][k] = np.copy(gamma)
                                
                            else:
                                if self.setup.main['verbose']:
                                    print('Covariance measure too high at',covMeasure,'; gamma is',gamma)
                                
                                # Increment gamma value (stabilization time)
                                gamma += 1
                                
                                # Determine resulting covariance after wait
                                _, _, _, covStabilize, covMeasure = \
                                    kalmanFilter(self.model[delta_lowest], covStabilize)                            
                    
                # Print normal row in table
                if self.setup.main['verbose']:
                    tab.print_row([delta, k, 'Measure '+
                       str(np.round(self.km[delta]['cov_tilde_measure'][k+delta], decimals=3))])
                
            # Delete iterable variable
            del k
        del delta
        
        
    
    def defActions(self):
        '''
        Define the actions of the finite-state abstraction (performed once,
        outside the iterative scheme).

        Returns
        -------
        None.

        '''
        
        Abstraction.defineActions(self)
    
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
                self.mdp.writePRISM_explicit(self.abstr, self.trans, self.setup.mdp['mode'])   
        
        else:
        
            # Create PRISM file (default way)
            self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
                self.mdp.writePRISM_scenario(self.abstr, self.trans, self.setup.mdp['mode'])  
              
            model_size = {'States':None,'Choices':None,'Transitions':None}

        self.time['4_MDPcreated'] = tocDiff(False)
        print('MDP created - time:',self.time['4_MDPcreated'])
        
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
        
    def preparePlots(self):
        '''
        Initializing function to prepare for plotting

        Returns
        -------
        None.

        '''
        
        # Process results
        self.plot           = dict()
    
        for delta_idx, delta in enumerate(self.setup.deltas):
            self.plot[delta] = dict()
            self.plot[delta]['N'] = dict()
            self.plot[delta]['T'] = dict()
            
            self.plot[delta]['N']['start'] = 0
            
            # Convert index to absolute time (note: index 0 is time tau)
            self.plot[delta]['T']['start'] = \
                int(self.plot[delta]['N']['start'] * self.basemodel.tau)
        
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
        
        import pandas as pd
        
        self.results = dict()
        
        policy_all = pd.read_csv(policy_file, header=None).iloc[:, 1:].fillna(-1).to_numpy()
        # Flip policy upside down (PRISM generates last time step at top!)
        policy_all = np.flipud(policy_all)
        
        policy_all = extractRowBlockDiag(policy_all, self.abstr['nr_regions'])
        
        self.results['optimal_policy'] = np.zeros(np.shape(policy_all))
        self.results['optimal_delta'] = np.zeros(np.shape(policy_all))
        self.results['optimal_reward'] = np.zeros(np.shape(policy_all))
        
        rewards_k0 = pd.read_csv(vector_file, header=None).iloc[1:].to_numpy()
        
        reward_rows = int((len(rewards_k0)-1)/self.abstr['nr_regions'] + 1)
        
        self.results['optimal_reward'][0:reward_rows, :] = \
            np.reshape(rewards_k0, (reward_rows, self.abstr['nr_regions']))
        
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
        
        if self.abstr['nr_regions'] <= 1000:
        
            from .postprocessing.createPlots import createProbabilityPlots
            
            if self.setup.plotting['probabilityPlots']:
                createProbabilityPlots(self.setup, self.plot[delta_value], 
                                       self.N, self.model[delta_value],
                                       self.results, self.abstr, self.mc)
                    
        else:
            
            printWarning("Omit probability plots (nr. of regions too large)")
            
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
        if self.setup.scenarios['gaussian'] is True:
            w_array = dict()
            for delta in self.setup.deltas:
                w_array[delta] = np.random.multivariate_normal(
                    np.zeros(self.model[delta].n), self.model[delta].noise['w_cov'],
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
                        
                        x = np.zeros((self.N+1, self.basemodel.n))
                        x_goal = [None]*(self.N+1)
                        x_region = np.zeros(self.N+1).astype(int)
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
                            
                            # Compute all centers of regions associated with points
                            x_cubic = skew2cubic(x[k], self.abstr)
                            
                            cubic_center = computeRegionCenters(x_cubic, 
                                                    self.basemodel.setup['partition']).flatten()
                            
                            if tuple(cubic_center) in self.abstr['allCentersCubic']:
                                # Save that state is currently in region ii
                                x_region[k] = self.abstr['allCentersCubic'][tuple(cubic_center)]
                                
                                # Retreive the action from the policy
                                actionToTake[k] = self.results['optimal_policy'][k, x_region[k]]
                            else:
                                x_region[k] = -1
                            
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
                                u[k] = np.array(self.model[delta].B_pinv @ ( x_goal[k+delta] - self.model[delta].A @ x[k] - self.model[delta].Q_flat ))
                                
                                # Implement the control into the physical (unobservable) system
                                x_hat = self.model[delta].A @ x[k] + self.model[delta].B @ u[k] + self.model[delta].Q_flat
                                
                                if self.setup.scenarios['gaussian'] is True:
                                    # Use Gaussian process noise
                                    x[k+delta] = x_hat + w_array[delta][n0id, i_abs, m, k]
                                else:
                                    # Use generated samples                                    
                                    disturbance = random.choice(self.model[delta].noise['samples'])
                                    
                                    x[k+delta] = x_hat + disturbance
                                    
                                # Add current state to trace
                                self.mc['traces'][n0][i][m] += [x[k+delta]]
                            
                            # Increase iterator variable by the value of delta associated to chosen action
                            k += delta
                        
                # Set of monte carlo iterations completed for specific initial state
                
                # Calculate the overall reachability probability for initial state i
                self.mc['results']['reachability_probability'][i,n0id] = \
                    np.sum(self.mc['results'][n0][i]['goalReached']) / self.setup.montecarlo['iterations']
                    
        self.time['6_MonteCarlo'] = tocDiff(False)
        print('Monte Carlo simulations finished:',self.time['6_MonteCarlo'])