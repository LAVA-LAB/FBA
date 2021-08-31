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
import pandas as pd

from .commons import writeFile, printSuccess

class mdp(object):
    def __init__(self, setup, N, abstr):
        '''
        Create the MDP model in Python for the current instance.
    
        Parameters
        ----------
        setup : dict
            Setup dictionary.
        N : int
            Finite time horizon.
        abstr : dict
            Abstraction dictionary
    
        Returns
        -------
        mdp : dict
            MDP dictionary.
    
        '''
        
        self.setup = setup
        self.N = N
        
        self.nr_regions = len(abstr['P'])
        self.nr_actions = abstr['nr_actions']
    
    def writePRISM_specification(self, mode, horizon):
        '''
        Write the PRISM specification / property to a file

        Parameters
        ----------
        mode : str
            Is either 'estimate' or 'interval'.
        horizon : str
            Is either 'finite' or 'infinite'.

        Returns
        -------
        specfile : str
            The name of the file in which the specification is stored.
        specification : str
            The specification itself given as a string.

        '''
        
        if horizon == 'infinite':
            # Infer number of time steps in horizon (at minimum delta value)
            self.horizonLen = int(self.N) #/self.setup.divide)
        elif horizon == 'steadystate':
            # Infer number of time steps in horizon (at minimum delta value)
            self.horizonLen = int(self.N) #/self.setup.divide)
        else:
            # We always need to provide a maximum nr. of steps, so make equal
            # to time horizon (even though horizon is also in the state space.)
            self.horizonLen = int(self.N) #/self.setup.divide)
            
        if mode == 'estimate':
            # If mode is default, set maximum probability as specification
            specification = 'Pmax=? [ F<='+str(self.horizonLen)+' "reached" ]'
            
        elif mode == 'interval':
            # If mode is interval, set lower bound of maximum prob. as spec.
            specification = 'Pmaxmin=? [ F<='+str(self.horizonLen)+' "reached" ]'
            
        # Define specification file
        specfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".pctl"
    
        # Write specification file
        writeFile(specfile, 'w', specification)
        
        return specfile, specification
    
    def writePRISM_explicit(self, abstr, trans, mode='estimate', km=None):
        '''
        Converts the model to the PRISM language, and write the model in
        explicit form to files (meaning that every transition is already
        enumerated explicitly).
    
        Parameters
        ----------
        abstr : dict
            Abstraction dictionary
        trans : dict
            Dictionary of transition probabilities        
        mode : str, optional
            Is either 'estimate' or 'interval'.
    
        Returns
        -------
        None.
    
        '''
        
        if self.setup.main['mode'] == 'Filter':
            if self.setup.mdp['k_steady_state'] != None:
                horizon = 'steadystate'
                region_reps_base = self.setup.mdp['k_steady_state'] + 1
                
            else:
                horizon = 'finite'
                region_reps_base = self.N
        else:
            horizon = 'infinite'
            region_reps_base = 1
            
        region_list = [[] for i in self.setup.all_deltas]
        k_list      = [[] for i in self.setup.all_deltas]
        k_max_list  = [[] for i in self.setup.all_deltas]
        delta_list  = [[] for i in self.setup.all_deltas]
        stoch_list  = [[] for i in self.setup.all_deltas]
        k_id_start  = [[] for i in self.setup.all_deltas]
            
        delta_start_id = {}
        
        for idx,delta in enumerate(self.setup.all_deltas):
            # Repetitions for current delta
            if delta == 1:
                region_reps = region_reps_base
                
            else:
                region_reps = km['waiting_time'] + delta
            
            length_before = np.sum([ len(i) for i in region_list ])
            delta_start_id[delta] = length_before
            
            # Region list for current delta
            region_list[idx] = np.tile( np.arange(self.nr_regions), region_reps )
            
            # Time step list
            k_list[idx] = np.repeat( np.maximum(np.arange(region_reps) - (delta-1), 0), self.nr_regions )
            
            # Maximum time step within this part of the model
            k_max_list[idx] = np.full( region_reps * self.nr_regions, fill_value = region_reps-1-(delta-1) )
            
            # Delta value list
            delta_list[idx] = np.full(region_reps * self.nr_regions, fill_value = delta)
            
            # Boolean whether transitions from this state are stochastic
            if delta == 1:
                stoch_list[idx] = np.full(region_reps * self.nr_regions, fill_value = True)
                
            else:
                stoch_list[idx] = np.concatenate((
                        np.full(self.nr_regions * (delta - 1), fill_value = False),
                        np.full(self.nr_regions * (km['waiting_time'] + 1), fill_value = True),
                    ))
                
            k_id_start[idx] = np.repeat([ length_before + i*self.nr_regions for i in range(region_reps) ], self.nr_regions)
        
        self.nr_states_per_delta = np.array([len(i) for i in region_list])
        self.nr_states = np.sum(self.nr_states_per_delta)
        
        # Compute how many states are added to the top of the transition list
        nr_extra_goal_states = max(self.setup.all_deltas)-1
        self.overhead = 3 + nr_extra_goal_states
        
        DF_DATA = {
            'state': np.arange(self.nr_states) + self.overhead,
            'region': [item for sublist in region_list for item in sublist],
            'delta': [item for sublist in delta_list for item in sublist], 
            'stoch': [item for sublist in stoch_list for item in sublist],
            'k': [item for sublist in k_list for item in sublist],
            'k_id_start': [item for sublist in k_id_start for item in sublist],
            'k_max': [item for sublist in k_max_list for item in sublist],
            }
        
        # print(len(DF_DATA['state']))
        # print(len(DF_DATA['region']))
        # print(len(DF_DATA['delta']))
        # print(len(DF_DATA['stoch']))
        # print(len(DF_DATA['k']))
        # print(len(DF_DATA['k_id_start']))
        # print(len(DF_DATA['k_max']))
        
        self.MAIN_DF = pd.DataFrame(data = DF_DATA)
        
        # Define PRISM filename
        PRISM_allfiles = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".all"
        
        print(' --- Writing PRISM states file')
        
        ### Write states file
        PRISM_statefile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".sta"
        
        state_file_string = '\n'.join(['(x)\n0:(-3)\n1:(-2)\n2:(-1)'] + 
            [str(3+i)+':('+str(-4-i)+')' for i in range(nr_extra_goal_states)] +
            [str(i+3+nr_extra_goal_states)+':('+str(i)+')' for i in range(self.nr_states)])
        
        # Write content to file
        writeFile(PRISM_statefile, 'w', state_file_string)
        
        print(' --- Writing PRISM label file')
        
        ### Write label file
        PRISM_labelfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".lab"
            
        label_file_list = ['0="init" 1="deadlock" 2="reached"'] + \
                          ['0: 1'] + \
                          ['1: 1'] + \
                          ['2: 2'] + \
                          ['' for i in range(self.nr_states)]
        
        for index, row in self.MAIN_DF.iterrows():
            
            # Write default string
            string = str(row['state'])+': 0'
            
            # Check if region is a deadlock state
            if len(abstr['actions'][row['delta']][row['region']]) == 0:
                string += ' 1'
            
            # Write current row
            label_file_list[index+3+1] = string
            
        label_file_string = '\n'.join(label_file_list)
           
        # Write content to file
        writeFile(PRISM_labelfile, 'w', label_file_string)
        
        print(' --- Writing PRISM transition file')
        
        ### Write transition file
        PRISM_transitionfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".tra"
            
        transition_file_list = ['' for i in range(len(self.MAIN_DF))]
            
        nr_choices_absolute = 0
        nr_transitions_absolute = 0
        
        printEvery = min(100, max(1, int(self.nr_states/10)))
        
        # For every row of the DataFrame
        for index, row in self.MAIN_DF.iterrows():
        
            state = row['state']
            r = row['region']
            
            if index % printEvery == 0:
                print(' ---- Write for state',index,'region',r,'time step',row['k'],'delta',row['delta'])
                
            choice = 0
            
            # Some rows are non-stochastic, because they only model an
            # arbitrary additional cost (because of the higher delta action)
            if not row['stoch']:
                
                # No actions enabled, so only add self-loop
                if mode == 'interval':
                    selfloop_prob = '[1.0,1.0] step'
                else:
                    selfloop_prob = '1.0 step'
                
                state_prime = state + self.nr_regions
                df_row_string = [[str(state) +' 0 '+str(state_prime)+' '+selfloop_prob]]
             
                # Increment choices and transitions both by one
                nr_choices_absolute += 1
                nr_transitions_absolute += 1
                choice += 1
            
            
            # If there exists an action in this state...
            elif all([len(abstr['actions'][delta][r]) == 0 for delta in self.setup.all_deltas]):        

                # No actions enabled, so only add self-loop
                if mode == 'interval':
                    selfloop_prob = '[1.0,1.0]'
                else:
                    selfloop_prob = '1.0'
                    
                if row['k'] >= self.N:
                    # If time horizon reached, to absorbing state
                    df_row_string = [[str(state) +' 0 0 '+selfloop_prob]]
                else:
                    # If not horizon reached, self-loop
                    df_row_string = [[str(state) +' 0 '+str(state)+' '+selfloop_prob]]
                
                # Increment choices and transitions both by one
                nr_choices_absolute += 1
                nr_transitions_absolute += 1
                choice += 1
                
                
            else:
                
                # If base rate, there's a normal action choice
                df_row_string = ['' for i in self.setup.all_deltas]  
                
                if row['delta'] == 1 or row['k'] == row['k_max']:          
                    allowed_deltas = self.setup.all_deltas  
          
                else:
                    allowed_deltas = [1]
          
                # For every delta value
                for delta_idx,delta in enumerate(allowed_deltas):
                    
                    action_string = ['' for i in range(len(abstr['actions'][delta][r]))]
                    
                    for a_idx,a in enumerate(abstr['actions'][delta][r]):
                        
                        string_start = str(state) +' '+ str(choice)
                        
                        if delta != 1:
                            # Jump to higher delta rate
                            succ_rep_start = delta_start_id[delta]
                            k_prime = 0
                            
                        elif row['k'] == row['k_max'] and row['delta'] != 1:
                            # Return to base rate
                            succ_rep_start = km['return_step'][row['delta']] * self.nr_regions
                            k_prime = km['return_step'][row['delta']]
                            
                        elif row['k'] == row['k_max'] and row['delta'] == 1:
                            # Loop to own iteration, since max. is reached
                            succ_rep_start = row['k_id_start']
                            k_prime = row['k']
                        
                        else:
                            # Proceed according to base rate
                            succ_rep_start = row['k_id_start'] + 1*self.nr_regions
                            k_prime = row['k'] + 1
                            
                        
                        # print('call action for delta',delta,'k_prime',k_prime,'action',a)
                        
                        # Call function to write transitions for this S-A pair
                        action_string[a_idx], transitions_plus = \
                            writePRISMtrans(trans, delta, k_prime, a, mode,
                                              string_start, succ_rep_start,
                                              self.overhead)
                            
                        # Increase choice counter
                        choice += 1
                        nr_choices_absolute += 1
                        nr_transitions_absolute += transitions_plus
                        
                    df_row_string[delta_idx] = action_string
                 
            transition_file_list[index] = df_row_string
                 
        self.dump = transition_file_list
            
        flatten = lambda t: [item for sublist in t for subsublist in sublist for item in subsublist]
        transition_file_flat = '\n'.join(flatten(transition_file_list))
        
        print(' ---- String ready; write to file...')
        
        # Header contains nr of states, choices, and transitions
        size_states = self.nr_states + self.overhead
        size_choices = nr_choices_absolute + self.overhead
        size_transitions = nr_transitions_absolute + self.overhead
        model_size = {'States': size_states, 
                      'Choices': size_choices, 
                      'Transitions':size_transitions}
        header = str(size_states)+' '+str(size_choices)+' '+str(size_transitions)+'\n'
        
        if mode == 'interval':
            firstrow = '0 0 0 [1.0,1.0]\n1 0 1 [1.0,1.0]\n2 0 2 [1.0,1.0]\n' + \
                '\n'.join([str(3+i)+
                    ' 0 '+str(3+i-1)+
                    ' [1.0,1.0]' for i in range(nr_extra_goal_states)]) + '\n'
        else:
            firstrow = '0 0 0 1.0\n1 0 1 1.0\n2 0 2 1.0\n' + \
                '\n'.join([str(3+i)+
                    ' 0 '+str(3+i-1)+
                    ' 1.0' for i in range(nr_extra_goal_states)]) + '\n'
        
        # Write content to file
        writeFile(PRISM_transitionfile, 'w', header+firstrow+transition_file_flat)
            
        ### Write specification file
        specfile, specification = self.writePRISM_specification(mode, horizon=horizon)
        
        return model_size, PRISM_allfiles, specfile, specification
        
   
        
def writePRISMtrans(trans, delta, k_prime, a, mode, string_start, succ_rep_start, overhead):
    
    # Define name of action
    actionLabel = str(a)+"_"+str(delta)+"_"+str(succ_rep_start)
    
    # print('Delta:',delta,'k_prime:',k_prime,'action',a)
    
    if mode == 'interval':
                        
        # Add probability to end in absorbing state
        deadlock_string = trans['prob'][delta][k_prime][a]['deadlock_interval_string']
        
        # Add probability to reach the goal
        goal_string = trans['prob'][delta][k_prime][a]['goal_interval_string']
        
        # Add probability to reach the goal
        critical_string = trans['prob'][delta][k_prime][a]['critical_interval_string']
    
        # Retreive probability intervals (and corresponding state ID's)
        probability_strings = trans['prob'][delta][k_prime][a]['interval_strings']
        probability_idxs    = trans['prob'][delta][k_prime][a]['interval_idxs']
    
        # Absorbing state has index zero
        action_string_a = [string_start+' 0 '+deadlock_string+' '+actionLabel]
        
        if goal_string != None:
            # Goal state has index 2
            goal_state_idx = 2 + delta - 1
            action_string_a += [string_start+' '+str(goal_state_idx)+' '+goal_string+' '+actionLabel]
            
        if critical_string != None:
            # Critical state has index 1
            action_string_a += [string_start+' 1 '+critical_string+' '+actionLabel]
        
        # Add resulting entries to the list
        action_string_b = [string_start +" "+str(i+overhead+succ_rep_start)+" "+intv+" "+actionLabel 
                      for (i,intv) in zip(probability_idxs,probability_strings) if intv]
        
    else:
        
        # Add probability to end in absorbing state
        deadlock_string = str(trans['prob'][delta][k_prime][a]['deadlock_approx'])
        
        # Add probability to reach the goal
        goal_string = str(trans['prob'][delta][k_prime][a]['goal_approx'])
        
        # Add probability to reach the goal
        critical_string = str(trans['prob'][delta][k_prime][a]['goal_approx'])
        
        # Retreive probability intervals (and corresponding state ID's)
        probability_strings = trans['prob'][delta][k_prime][a]['approx_strings']
        probability_idxs    = trans['prob'][delta][k_prime][a]['approx_idxs']
        
        if float(deadlock_string) > 0:
            # Absorbing state has index zero
            action_string_a = [string_start+' 0 '+deadlock_string+' '+actionLabel]
        else:
            action_string_a = []
            
        if float(goal_string) > 0:
            # Goal state has index 2
            goal_state_idx = 2 + delta - 1
            action_string_a += [string_start+' '+str(goal_state_idx)+' '+goal_string+' '+actionLabel]
            
        if float(critical_string) > 0:
            # Critical state has index 1
            action_string_a += [string_start+' 1 '+critical_string+' '+actionLabel]
            
        # Add resulting entries to the list
        action_string_b = [string_start +" "+str(i+overhead+succ_rep_start)+" "+intv+" "+actionLabel 
                      for (i,intv) in zip(probability_idxs,probability_strings) if intv]
        
    nr_transitions = len(action_string_a) + len(action_string_b)
    
    merged_string = '\n'.join(action_string_a + action_string_b)
    
    return merged_string, nr_transitions
        
