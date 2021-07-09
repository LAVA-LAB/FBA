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

from .mainFunctions import computeRegionCenters, cubic2skew, skew2cubic
from .commons import tic, ticDiff, tocDiff, table, printWarning, floor_decimal

from .createMDP import mdp

from .abstraction import Abstraction

class scenarioBasedAbstraction(Abstraction):
    def __init__(self, setup, system):
        '''
        Initialize scenario-based abstraction (ScAb) object

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
        self.km = None
        
        Abstraction.__init__(self)
        
        # Define state space partition
        Abstraction.definePartition(self)
        
    def _loadScenarioTable(self, tableFile):
        '''
        Load tabulated bounds on the transition probabilities (computed using
        the scenario approach).

        Parameters
        ----------
        tableFile : str
            File from which to load the table.

        Returns
        -------
        memory : dict
            Dictionary containing all loaded probability bounds / intervals.

        '''
        
        memory = dict()
        
        if not os.path.isfile(tableFile):
            sys.exit('ERROR: the following table file does not exist:'+
                     str(tableFile))
        
        with open(tableFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i,row in enumerate(reader):
                
                strSplit = row[0].split(',')
                
                key = tuple( [int(float(i)) for i in strSplit[0:2]] + 
                             [float(strSplit[2])] )
                
                value = [float(i) for i in strSplit[-2:]]
                memory[key] = value
                    
        return memory
        
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
        
        prob = dict()

        printEvery = min(100, max(1, int(self.abstr['nr_actions']/10)))

        # For every action (i.e. target point)
        for a in range(self.abstr['nr_actions']):
            
            # Check if action a is available in any state at all
            if len(self.abstr['actions_inv'][delta][a]) > 0:
                    
                prob[a] = dict()
            
                mu = self.abstr['target']['d'][a]
                Sigma = self.model[delta]['noise']['w_cov']
                
                if self.setup.scenarios['gaussian'] is True:
                    # Compute Gaussian noise samples
                    samples = np.random.multivariate_normal(mu, Sigma, 
                                    size=self.setup.scenarios['samples'])
                    
                else:
                    # Determine non-Gaussian noise samples (relative from 
                    # target point)
                    samples = mu + np.array(
                        random.choices(self.model[delta]['noise']['samples'], 
                        k=self.setup.scenarios['samples']) )
                    
                # Transform samples back to hypercubic partition
                samplesCubic = skew2cubic(samples, self.abstr)
                    
                prob[a] = self._scenarioTransition(self.setup, 
                      self.system.partition, 
                      self.abstr, self.trans, samplesCubic)
                
                # Print normal row in table
                if a % printEvery == 0:
                    nr_transitions = len(prob[a]['interval_idxs'])
                    tab.print_row([delta, k, a, 
                       'Probabilities computed (transitions: '+
                       str(nr_transitions)+')'])
                
        return prob
    
    def _scenarioTransition(setup, partition, abstr, trans, samples):
        '''
        Compute the transition probability intervals
    
        Parameters
        ----------
        setup : dict
            Setup dictionary.
        partition : dict
            Dictionary of the partition.
        abstr : dict
            Dictionay containing all information of the finite-state abstraction.
        trans : dict
            Dictionary with all data for the transition probabilities
        samples : 2D Numpy array
            Numpy array, with every row being a sample of the process noise.
    
        Returns
        -------
        returnDict : dict
            Dictionary with the computed (intervals of) transition probabilities.
    
        '''
        
        # Number of decision variables always equal to one
        d = 1
        Nsamples = setup.scenarios['samples']
        beta = setup.scenarios['confidence']
        
        # Initialize counts array
        counts = dict()
        
        # Compute to which regions the samples belong
        centers_cubic = computeRegionCenters(samples, partition)
        
        for s in range(Nsamples):
            
            key = tuple(centers_cubic[s])
            
            if key in abstr['allCentersCubic']:
                
                idx = abstr['allCentersCubic'][ key ]
                
                if idx in counts:
                    counts[idx] += 1
                else:
                    counts[idx] = 1
        
        # Count number of samples not in any region (i.e. in absorbing state)
        k = Nsamples - sum(counts.values()) + d
    
        key_lb = tuple( [Nsamples, k, beta] )
        key_ub = tuple( [Nsamples, k-1, beta] ) 
        
        deadlock_low = trans['memory'][key_ub][0]
        if k > Nsamples:
            deadlock_upp = 1
        else:
            deadlock_upp = trans['memory'][key_lb][1]
    
        # Initialize vectors for probability bounds
        probability_low = np.zeros(len(counts))
        probability_upp = np.zeros(len(counts))
        probability_approx = np.zeros(len(counts))
        
        interval_idxs = np.zeros(len(counts), dtype=int)
        approx_idxs = np.zeros(len(counts), dtype=int)
    
        # Enumerate over all the non-zero bins
        for i, (region,count) in enumerate(counts.items()): #zip(np.arange(abstr['nr_regions'])[nonEmpty], counts[nonEmpty]):
            
            k = Nsamples - count + d
            
            key_lb = tuple( [Nsamples, k,   beta] )
            key_ub = tuple( [Nsamples, k-1, beta] )
            
            if k > Nsamples:
                probability_low[i] = 0                
            else:
                probability_low[i] = 1 - trans['memory'][key_lb][1]
            probability_upp[i] = 1 - trans['memory'][key_ub][0]
            
            interval_idxs[i] = int(region)
            
            # Point estimate transition probability (count / total)
            probability_approx[i] = count / Nsamples
            approx_idxs[i] = int(region)
        
        nr_decimals = 5
        
        # PROBABILITY INTERVALS
        probs_lb = floor_decimal(probability_low, nr_decimals)
        probs_ub = floor_decimal(probability_upp, nr_decimals)
        
        # Create interval strings (only entries for prob > 0)
        interval_strings = ["["+
                          str(floor_decimal(max(1e-4, lb),nr_decimals))+","+
                          str(floor_decimal(min(1,    ub),nr_decimals))+"]"
                          for (lb, ub) in zip(probs_lb, probs_ub)]# if ub > 0]
        
        # Compute deadlock probability intervals
        deadlock_lb = floor_decimal(deadlock_low, nr_decimals)
        deadlock_ub = floor_decimal(deadlock_upp, nr_decimals)
        
        deadlock_string = '['+ \
                           str(floor_decimal(max(1e-4, deadlock_lb),nr_decimals))+','+ \
                           str(floor_decimal(min(1,    deadlock_ub),nr_decimals))+']'
        
        # POINT ESTIMATE PROBABILITIES
        probability_approx = np.round(probability_approx, nr_decimals)
        
        # Create approximate prob. strings (only entries for prob > 0)
        approx_strings = [str(p) for p in probability_approx]# if p > 0]
        
        # Compute approximate deadlock transition probabilities
        deadlock_approx = np.round(1-sum(probability_approx), nr_decimals)
        
        returnDict = {
            'interval_strings': interval_strings,
            'interval_idxs': interval_idxs,
            'approx_strings': approx_strings,
            'approx_idxs': approx_idxs,
            'deadlock_interval_string': deadlock_string,
            'deadlock_approx': deadlock_approx,
        }
        
        return returnDict
    
    # def defActions(self):
    #     '''
    #     Define the actions of the finite-state abstraction (performed once,
    #     outside the iterative scheme).

    #     Returns
    #     -------
    #     None.

    #     '''
        
    #     Abstraction.defineActions(self)
    
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
                
        print(' -- Loading scenario approach table...')
        
        tableFile = 'input/probabilityTable_N='+ \
                        str(self.setup.scenarios['samples'])+'_beta='+ \
                        str(self.setup.scenarios['confidence'])+'.csv'
        
        # Load scenario approach table
        self.trans['memory'] = self._loadScenarioTable(tableFile = tableFile)
        
        # Retreive type of horizon
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
                    print(' -- Calculate transitions for delta',
                          delta,'at time',k)
                
                self.trans['prob'][delta][k] = \
                    self._computeProbabilityBounds(tab, k, delta)
                
            # Delete iterable variables
            del k
        del delta
        
        self.time['3_probabilities'] = tocDiff(False)
        print('Transition probabilities calculated - time:',
              self.time['3_probabilities'])

    # def buildMDP(self):
    #     '''
    #     Build the (i)MDP and create all respective PRISM files.

    #     Returns
    #     -------
    #     model_size : dict
    #         Dictionary describing the number of states, choices, and 
    #         transitions.

    #     '''
        
    #     # Initialize MDP object
    #     self.mdp = mdp(self.setup, self.N, self.abstr)
        
    #     if self.setup.mdp['prism_model_writer'] == 'explicit':
            
    #         # Create PRISM file (explicit way)
    #         model_size, self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
    #             self.mdp.writePRISM_explicit(self.abstr, self.trans, mode=self.setup.mdp['mode'])   
        
    #     else:
        
    #         # Create PRISM file (default way)
    #         self.mdp.prism_file, self.mdp.spec_file, self.mdp.specification = \
    #             self.mdp.writePRISM_scenario(self.abstr, self.trans, self.setup.mdp['mode'])  
              
    #         model_size = {'States':None,'Choices':None,'Transitions':None}

    #     self.time['4_MDPcreated'] = tocDiff(False)
    #     print('MDP created - time:',self.time['4_MDPcreated'])
        
    #     return model_size
            
    # def solveMDP(self):
    #     '''
    #     Solve the (i)MDP usign PRISM

    #     Returns
    #     -------
    #     None.

    #     '''
            
    #     # Solve the MDP in PRISM (which is called via the terminal)
    #     policy_file, vector_file = self._solveMDPviaPRISM()
        
    #     # Load PRISM results back into Python
    #     self.loadPRISMresults(policy_file, vector_file)
            
    #     self.time['5_MDPsolved'] = tocDiff(False)
    #     print('MDP solved in',self.time['5_MDPsolved'])
        
    # def preparePlots(self):
    #     '''
    #     Initializing function to prepare for plotting

    #     Returns
    #     -------
    #     None.

    #     '''
        
    #     # Process results
    #     self.plot           = dict()
    
    #     for delta_idx, delta in enumerate(self.setup.deltas):
    #         self.plot[delta] = dict()
    #         self.plot[delta]['N'] = dict()
    #         self.plot[delta]['T'] = dict()
            
    #         self.plot[delta]['N']['start'] = 0
            
    #         # Convert index to absolute time (note: index 0 is time tau)
    #         self.plot[delta]['T']['start'] = \
    #             int(self.plot[delta]['N']['start'] * self.system.LTI['tau'])
        
    # def _solveMDPviaPRISM(self):
    #     '''
    #     Call PRISM to solve (i)MDP while executing the Python codes.

    #     Returns
    #     -------
    #     policy_file : str
    #         Name of the file in which the optimal policy is stored.
    #     vector_file : str
    #         Name of the file in which the optimal rewards are stored.

    #     '''
        
    #     import subprocess

    #     prism_folder = self.setup.mdp['prism_folder'] 
        
    #     print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
    #     print('Starting PRISM...')
        
    #     spec = self.mdp.specification
    #     mode = self.setup.mdp['mode']
    #     java_memory = self.setup.mdp['prism_java_memory']
        
    #     print(' -- Running PRISM with specification for mode',
    #           mode.upper()+'...')
    
    #     file_prefix = self.setup.directories['outputFcase'] + "PRISM_" + mode
    #     policy_file = file_prefix + '_policy.csv'
    #     vector_file = file_prefix + '_vector.csv'
    
    #     options = ' -ex -exportadv "'+policy_file+'"'+ \
    #               ' -exportvector "'+vector_file+'"'
    
    #     # Switch between PRISM command for explicit model vs. default model
    #     if self.setup.mdp['prism_model_writer'] == 'explicit':
    
    #         print(' --- Execute PRISM command for EXPLICIT model description')        
    
    #         model_file      = '"'+self.mdp.prism_file+'"'             
        
    #         # Explicit model
    #         command = prism_folder+"bin/prism -javamaxmem "+str(java_memory)+"g "+ \
    #                   "-importmodel "+model_file+" -pf '"+spec+"' "+options
    #     else:
            
    #         print(' --- Execute PRISM command for DEFAULT model description')
            
    #         model_file      = '"'+self.mdp.prism_file+'"'
            
    #         # Default model
    #         command = prism_folder+"bin/prism -javamaxmem "+str(java_memory)+"g "+ \
    #                   model_file+" -pf '"+spec+"' "+options    
        
    #     subprocess.Popen(command, shell=True).wait()    
        
    #     return policy_file, vector_file
        
    # def loadPRISMresults(self, policy_file, vector_file):
    #     '''
    #     Load results from existing PRISM output files.

    #     Parameters
    #     ----------
    #     policy_file : str
    #         Name of the file to load the optimal policy from.
    #     vector_file : str
    #         Name of the file to load the optimal policy from.

    #     Returns
    #     -------
    #     None.

    #     '''
        
    #     import pandas as pd
        
    #     self.results = dict()
        
    #     policy_all = pd.read_csv(policy_file, header=None).iloc[:, 1:].fillna(-1).to_numpy()
    #     # Flip policy upside down (PRISM generates last time step at top!)
    #     policy_all = np.flipud(policy_all)
        
    #     self.results['optimal_policy'] = np.zeros(np.shape(policy_all))
    #     self.results['optimal_delta'] = np.zeros(np.shape(policy_all))
    #     self.results['optimal_reward'] = np.zeros(np.shape(policy_all))
        
    #     rewards_k0 = pd.read_csv(vector_file, header=None).iloc[1:].to_numpy()
    #     self.results['optimal_reward'][0,:] = rewards_k0.flatten()
        
    #     # Split the optimal policy between delta and action itself
    #     for i,row in enumerate(policy_all):
            
    #         for j,value in enumerate(row):
                
    #             # If value is not -1 (means no action defined)
    #             if value != -1:
    #                 # Split string
    #                 value_split = value.split('_')
    #                 # Store action and delta value separately
    #                 self.results['optimal_policy'][i,j] = int(value_split[1])
    #                 self.results['optimal_delta'][i,j] = int(value_split[3])
    #             else:
    #                 # If no policy is known, set to -1
    #                 self.results['optimal_policy'][i,j] = int(value)
    #                 self.results['optimal_delta'][i,j] = int(value) 
        
    # def generatePlots(self, delta_value, max_delta):
    #     '''
    #     Generate (optimal reachability probability) plots

    #     Parameters
    #     ----------
    #     delta_value : int
    #         Value of delta for the model to plot for.
    #     max_delta : int
    #         Maximum value of delta used for any model.

    #     Returns
    #     -------
    #     None.
    #     '''
        
    #     print('\nGenerate plots')
        
    #     if len(self.abstr['P']) <= 1000:
        
    #         from .postprocessing.createPlots import createProbabilityPlots
            
    #         if self.setup.plotting['probabilityPlots']:
    #             createProbabilityPlots(self.setup, self.plot[delta_value], 
    #                                    self.N, self.model[delta_value],
    #                                    self.system.partition,
    #                                    self.results, self.abstr, self.mc)
                    
    #     else:
            
    #         printWarning("Omit probability plots (nr. of regions too large)")
            
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
            init_state_idxs = self.abstr['P'].keys() #np.arange(len(self.abstr['P']))
            
        else:
            init_state_idxs = self.setup.montecarlo['init_states']
        
        # The gaussian random variables are precomputed to speed up the code
        if self.setup.scenarios['gaussian'] is True:
            w_array = dict()
            for delta in self.setup.deltas:
                w_array[delta] = np.random.multivariate_normal(
                    np.zeros(self.model[delta]['n']), self.model[delta]['noise']['w_cov'],
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
                    
                    if i in self.abstr['goal'][n0]:
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
                        
                        x = np.zeros((self.N, self.system.LTI['n']))
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
                            
                            # Compute all centers of regions associated with points
                            x_cubic = skew2cubic(x[k], self.abstr)
                            
                            cubic_center = computeRegionCenters(x_cubic, 
                                                    self.system.partition).flatten()
                            
                            if tuple(cubic_center) in self.abstr['allCentersCubic']:
                                # Save that state is currently in region ii
                                x_region[k] = self.abstr['allCentersCubic'][tuple(cubic_center)]
                                
                                # Retreive the action from the policy
                                actionToTake[k] = self.results['optimal_policy'][k, x_region[k]]
                            else:
                                x_region[k] = -1
                            
                            # If current region is the goal state ... 
                            if x_region[k] in self.abstr['goal'][k]:
                                # Then abort the current iteration, as we have achieved the goal
                                self.mc['results'][n0][i]['goalReached'][m] = True
                                
                                if self.setup.main['verbose']:
                                    tab.print_row([n0, i, m, k, 'Goal state reached'], sort="Success")
                                break
                            # If current region is in critical states...
                            elif x_region[k] in self.abstr['critical'][k]:
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
                                u[k] = np.array(self.model[delta]['B_pinv'] @ ( x_goal[k+delta] - self.model[delta]['A'] @ x[k] - self.model[delta]['Q_flat'] ))
                                
                                # Implement the control into the physical (unobservable) system
                                x_hat = self.model[delta]['A'] @ x[k] + self.model[delta]['B'] @ u[k] + self.model[delta]['Q_flat']
                                
                                if self.setup.scenarios['gaussian'] is True:
                                    # Use Gaussian process noise
                                    x[k+delta] = x_hat + w_array[delta][n0id, i_abs, m, k]
                                else:
                                    # Use generated samples                                    
                                    disturbance = random.choice(self.model[delta]['noise']['samples'])
                                    
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