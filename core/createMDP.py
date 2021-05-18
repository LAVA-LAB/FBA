# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:47:22 2021

@author: Thom Badings
"""

import numpy as np

from .commons import floor_decimal, writeFile, \
                     printWarning, printSuccess

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
        kalman : dict
            Dictionary containing the Kalman filter dynamics.
        abstr : dict
            Dictionay containing all information of the finite-state abstraction.
    
        Returns
        -------
        mdp : dict
            MDP dictionary.
    
        '''
        
        self.setup = setup
        self.N = N
        
        self.nr_regions = abstr['nr_regions']
        self.nr_actions = abstr['nr_actions']
        
        if self.setup.mdp['horizon'] == 'finite':
        
            self.nr_states  = self.nr_regions * self.N
            
            # Specify goal state set
            self.goodStates = [g + x*self.nr_regions for x in range(N) for g in abstr['goal']]
            
            # Specify sets that can never reach target set
            self.badStates  = [c + x*self.nr_regions for x in range(N) for c in abstr['critical']]
            
        else:
            
            self.nr_states = self.nr_regions
            
            # Specify goal state set
            self.goodStates = abstr['goal']
            
            # Specify sets that can never reach target set
            self.badStates  = abstr['critical']
            
            
    def createPython(self, abstr, trans, kalman=False):
        
        # Create MDP internally in Python
        
        # self.nr_actions = 0
        
        self.transitions = dict()
        self.transitions['prob'] = dict()
        self.transitions['succ'] = dict()
        
        min_delta = min(self.setup.deltas)

        if self.setup.scenarios['switch']:
            printWarning('Warning: probabilities based on average between bounds; results may be unrealistic')
    
        # For every time step in the horizon
        for k in range(self.N):
            self.transitions['prob'][k] = dict()
            self.transitions['succ'][k] = dict()
    
            # For every origin state
            for s_rel in range(self.nr_regions):
                
                s = s_rel + k * self.nr_regions
                
                self.transitions['prob'][k][s] = dict()
                self.transitions['succ'][k][s] = dict()
                
                # For every delta value in the list
                for delta in self.setup.deltas:
                    
                    # Only proceed if we stay below the time horizon
                    if k + delta < self.N:
                    
                        # For every enaled action in state s, via delta
                        actions         = abstr['actions'][delta][s_rel]
                        
                        # For every enabled action
                        for action in actions:
                            
                            to_probs_all    = trans['prob'][delta][k][action]['approx']
                            
                            to_probs = to_probs_all #[ to_probs_all.nonzero() ]
                            to_states = np.arange(self.nr_regions) #[ to_probs_all.nonzero() ]
                            
                            if kalman is not False:
                                # Retreive required waiting time
                                gamma = kalman[delta]['waiting_time'][k]
                            else:
                                gamma = 0
                                LIC = False
                            
                            if LIC and gamma > 0:
                                # If LIC is active, take the waiting time into account
                                
                                # If waiting time is above zero, take into account
                                # Only proceed if waiting is indeed enabled
                                if abstr['waitEnabled'][min_delta][action]:
                                    
                                    # Compute index shift to include waiting time
                                    idx_shift = (k + delta) * self.nr_regions + \
                                        gamma * self.nr_regions * min_delta
                                    
                                    # Only add if the resulting successor is still
                                    # within the finite horizon                                    
                                    if k + delta + gamma*min_delta < self.N:                                
                                    
                                        self.transitions['prob'][k][s][(delta,action)] = to_probs
                                        self.transitions['succ'][k][s][(delta,action)] = to_states + idx_shift
                                        
                                        # Increment the number of actions
                                        # self.nr_actions += 1
        
                            else:
                                # If LIC is not active, directly add the transition
                                
                                # Compute index shift for pointing forward in time
                                idx_shift = (k + delta) * self.nr_regions
                                
                                 # Only add if the resulting successor is still
                                # within the finite horizon                                
                                self.transitions['prob'][k][s][(delta,action)] = to_probs
                                self.transitions['succ'][k][s][(delta,action)] = to_states + idx_shift
                                
                                # Increment the number of actions
                                # self.nr_actions += 1

    def writePRISM_scenario(self, abstr, trans, mode='estimate', horizon='infinite'):
        '''
        Converts the model to the PRISM language, and write the model to a file.
    
        Parameters
        ----------
        mode : str, optional
            Is either 'estimate' or 'interval'.
        horizon : str, optional
            Is either 'finite' or 'infinite'.
    
        Returns
        -------
        None.
    
        '''
    
        if horizon == 'infinite':
            min_delta = self.N
        else:
            min_delta = min(self.setup.deltas)                
    
        # Define PRISM filename
        PRISM_file = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".prism"
    
        if mode == "default":
            modeltype = "MDP"
        else:
            modeltype = "iMDP"
    
        # Write header for the PRISM file
        if horizon == 'infinite':
            header = [
                "// "+modeltype+" created by the filter-based abstraction method \n",
                "// Infinite horizon version \n\n"
                "mdp \n\n",
                # "const int xInit; \n\n",
                "const int regions = "+str(int(self.nr_regions-1))+"; \n\n",
                "module "+modeltype+"_abstraction \n\n",
                ]
            
            # Define variables in module
            variable_defs = [
                "\tx : [-1..regions]; \n\n",
                ]
            
        else:
            header = [
                "// "+modeltype+" created by the filter-based abstraction method \n\n",
                "mdp \n\n",
                # "const int xInit; \n\n",
                "const int Nhor = "+str(int(self.N / min_delta))+"; \n",
                "const int regions = "+str(int(self.nr_regions-1))+"; \n\n",
                "module iMDP \n\n",
                ]
        
            # Define variables in module
            variable_defs = [
                "\tk : [0..Nhor]; \n",
                "\tx : [-1..regions]; \n\n",
                ]
        
        # Add initial header to PRISM file
        writeFile(PRISM_file, 'w', header+variable_defs)
            
        #########
        
        # Define actions
        for k in range(0, self.N, min_delta):
            
            # Add comment to list
            if horizon == 'finite':
                action_defs = ["\t// Actions for k="+str(k)+"\n"]
                
            else:
                action_defs = []    
            
            # For every delta value in the list
            for delta_idx,delta in enumerate(self.setup.deltas):
                
                action_defs += ["\t// Delta="+str(delta)+"\n"]
                
                # Only continue if resulting time is below horizon
                boolean = True
                if k % delta != 0:
                    boolean = False
                # Always proceed if horizon is infinite
                if (k + delta <= self.N and boolean) or horizon == 'infinite':
                    
                    # For every action (i.e. target state)
                    for a in range(self.nr_actions):
                        
                        # Define name of action
                        actionLabel = "[a_"+str(a)+"_d_"+str(delta)+"]"
                        
                        # Retreive in which states this action is enabled
                        enabledIn = abstr['actions_inv'][delta][a]
                        
                        # Only continue if this action is enabled anywhere
                        if len(enabledIn) > 0:
                            
                            # Write guards
                            guardPieces = ["x="+str(i) for i in enabledIn]
                            sep = " | "
                            
                            if horizon == 'infinite':
                                # Join individual pieces and write full guard
                                guard = sep.join(guardPieces)
                                
                                # For infinite time horizon, kprime not used
                                kprime = ""
                                
                            else:                
                                # Join  individual pieces
                                guardStates = sep.join(guardPieces)
                                
                                # Write full guard
                                guard = "k="+str(int(k/min_delta))+" & ("+guardStates+")"
                                
                                # Compute successor state time step
                                kprime = "&(k'=k+"+str(int(delta/min_delta))+")"
                            
                            
                            
                            if self.setup.scenarios['switch'] and mode == 'interval':
                                
                                # Retreive probability intervals (and corresponding state ID's)
                                interval_idxs    = trans['prob'][delta][k][a]['interval_idxs']
                                interval_strings = trans['prob'][delta][k][a]['interval_strings']
                                
                                # If mode is interval, use intervals on probs.
                                succPieces = [intv+" : (x'="+
                                              str(i)+")"+kprime 
                                              for (i,intv) in zip(interval_idxs,interval_strings)]
                                
                                # Use absorbing state to make sure that probs sum to one
                                deadlock_string = trans['prob'][delta][k][a]['deadlock_interval_string']
                                succPieces += [deadlock_string+" : (x'=-1)"+kprime]
                                
                            else:
                                
                                if mode == 'estimate':
                                    # Write resulting states with their probabilities
                                    succProbStrings = trans['prob'][delta][k][a]['approx_strings']
                                    succProbIdxs    = trans['prob'][delta][k][a]['approx_idxs']
                                    
                                    # If mode is default, use concrete probabilities
                                    succPieces = [str(p)+":(x'="+str(i)+")"+kprime
                                                  for (i,p) in zip(succProbIdxs,succProbStrings)]
                                    
                                    # Use absorbing state to make sure that probs sum to one
                                    deadlockProb = trans['prob'][delta][k][a]['deadlock_approx']
                                    if float(deadlockProb) > 0:
                                        succPieces += [str(deadlockProb)+":(x'=-1)"+kprime]
                                    
                                else:
                                    # Write resulting states with their probabilities
                                    succProbs = trans['prob'][delta][k][a]['approx']
                                    
                                    # If mode is interval, use intervals on probs.
                                    succPieces = ["["+
                                                  str(floor_decimal(max(1e-4, p - self.setup.gaussian['margin']),5))+","+
                                                  str(floor_decimal(min(1, p + self.setup.gaussian['margin']),5))+"] : (x'="+
                                                  str(i)+")"+kprime 
                                                  for i,p in enumerate(succProbs) if p > 0]
                            
                                    # Use absorbing state to make sure that probs sum to one
                                    deadlock_lb = np.round(1-sum(succProbs),6) - self.setup.gaussian['margin']
                                    deadlock_ub = np.round(1-sum(succProbs),6) + self.setup.gaussian['margin']
                                    succPieces += ["["+
                                        str(floor_decimal(max(1e-4, deadlock_lb),5))+","+
                                        str(floor_decimal(min(1,    deadlock_ub),5))+"] : (x'=-1)"+kprime]
                                
                            # Join  individual pieces
                            sep = " + "
                            successors = sep.join(succPieces)
                            
                            # Join pieces to write full action definition
                            action_defs += "\t"+actionLabel+" " + guard + \
                                " -> " + successors + "; \n"
            
            # Insert extra white lines
            action_defs += ["\n\n"]
            
            # Add actions to PRISM file
            writeFile(PRISM_file, "a", action_defs)
            
        #########
        
        if horizon == 'infinite':
            footer = [
                "endmodule \n\n",
                "init x > -1 endinit \n\n"
                ]
            
        else:
            footer = [
                "endmodule \n\n",
                "init k=0 endinit \n\n"
                ]
        
        labelPieces = ["(x="+str(x)+")" for x in abstr['goal']]
        sep = "|"
        labelGuard = sep.join(labelPieces)
        labels = [
            "// Labels \n",
            "label \"reached\" = "+labelGuard+"; \n"
            ]
        
        # Add footer and label definitions to PRISM file
        writeFile(PRISM_file, "a", footer + labels)
        
        #########
        specfile, specification = self.writePRISM_specification(mode, horizon)
        
        # # Define terminal command (may be user-dependent)
        # commandfile = self.setup.directories['outputFcase']+ \
        #     self.setup.mdp['filename']+"_"+mode+"_command.txt"
            
        # # Write command file
        # command = "bin/prism models/m2/Abstraction_interval.prism -pf 'Pmaxmin=?[F<="+\
        #     str(horizonLen)+" \"reached\"]' -exportadv strat.csv -exportstates states.csv -exportvector vector.csv"
        # writeFile(commandfile, 'w', command)
        
        if mode == 'estimate':
            printSuccess('MDP ('+horizon+' horizon) exported as PRISM file')
        else:
            printSuccess('iMDP ('+horizon+' horizon) exported as PRISM file')
        
        return PRISM_file, specfile, specification
    
    def writePRISM_specification(self, mode, horizon):
        
        if horizon == 'infinite':
            # Infer number of time steps in horizon (at minimum delta value)
            horizonLen = int(self.N/min(self.setup.deltas))
            
            if mode == 'estimate':
                # If mode is default, set maximum probability as specification
                specification = 'Pmax=? [ F<='+str(horizonLen)+' "reached" ]'
                
            elif mode == 'interval':
                # If mode is interval, set lower bound of maximum prob. as spec.
                specification = 'Pmaxmin=? [ F<='+str(horizonLen)+' "reached" ]'
            
        else:
            if mode == 'estimate':
                # If mode is default, set maximum probability as specification
                specification = 'Pmax=? [ F "reached" ]'
                
            elif mode == 'interval':
                # If mode is interval, set lower bound of maximum prob. as spec.
                specification = 'Pmaxmin=? [ F "reached" ]'
            
        # Define specification file
        specfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".pctl"
    
        # Write specification file
        writeFile(specfile, 'w', specification)
        
        return specfile, specification
    
    def writePRISM_explicit(self, abstr, trans, mode='estimate'):
        
        # Define PRISM filename
        PRISM_allfiles = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".all"
        
        print(' --- Writing PRISM states file')
        
        ### Write states file
        PRISM_statefile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".sta"
        
        state_file_string = '\n'.join(['(x)\n0:(-1)'] + [str(i+1)+':('+str(i)+')' for i in range(self.nr_regions)])
        
        # Write content to file
        writeFile(PRISM_statefile, 'w', state_file_string)
        
        print(' --- Writing PRISM label file')
        
        ### Write label file
        PRISM_labelfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".lab"
            
        label_file_list = ['0="init" 1="deadlock" 2="reached"'] + \
                          ['0: 1'] + \
                          ['' for i in range(self.nr_regions)]
        
        for i in range(self.nr_regions):
            substring = str(i+1)+': 0'
            
            # Check if region is a deadlock state
            for delta in self.setup.deltas:
                if len(abstr['actions'][delta][i]) == 0:
                    substring += ' 1'
                    break
            
            # Check if region is in goal set
            if i in self.goodStates:
                substring += ' 2'
            
            label_file_list[i+2] = substring
            
        label_file_string = '\n'.join(label_file_list)
           
        # Write content to file
        writeFile(PRISM_labelfile, 'w', label_file_string)
        
        print(' --- Writing PRISM transition file')
        
        ### Write transition file
        PRISM_transitionfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".tra"
            
        transition_file_list = ['' for i in range(self.nr_regions)]
            
        nr_choices_absolute = 0
        nr_transitions_absolute = 0
        
        printEvery = min(100, max(1, int(self.nr_regions/10)))
        
        # For every state
        for s in range(self.nr_regions):
            
            
            if s % printEvery == 0:
                print(' ---- Write for region',s)
            
            substring = ['' for i in range(len(self.setup.deltas))]
            
            choice = 0
            selfloop = False
            
            # Check if region is a deadlock state
            for delta_idx,delta in enumerate(self.setup.deltas):    
            
                if len(abstr['actions'][delta][s]) > 0:
            
                    subsubstring = ['' for i in range(len(abstr['actions'][delta][s]))]
            
                    # For every enabled action                
                    for a_idx,a in enumerate(abstr['actions'][delta][s]):
                        
                        # Define name of action
                        actionLabel = "a_"+str(a)+"_d_"+str(delta)
                        
                        substring_start = str(s+1) +' '+ str(choice)
                        
                        if self.setup.scenarios['switch'] and mode == 'interval':
                        
                            # Add probability to end in absorbing state
                            deadlock_string = trans['prob'][delta][0][a]['deadlock_interval_string']
                        
                            # Retreive probability intervals (and corresponding state ID's)
                            probability_strings = trans['prob'][delta][0][a]['interval_strings']
                            probability_idxs    = trans['prob'][delta][0][a]['interval_idxs']
                        
                            # Absorbing state has index zero
                            subsubstring_a = [substring_start+' 0 '+deadlock_string+' '+actionLabel]
                            
                            # Add resulting entries to the list
                            subsubstring_b = [substring_start +" "+str(i+1)+" "+intv+" "+actionLabel 
                                          for (i,intv) in zip(probability_idxs,probability_strings) if intv]
                            
                        else:
                            
                            # Add probability to end in absorbing state
                            deadlock_string = str(trans['prob'][delta][0][a]['deadlock_approx'])
                            
                            # Retreive probability intervals (and corresponding state ID's)
                            probability_strings = trans['prob'][delta][0][a]['approx_strings']
                            probability_idxs    = trans['prob'][delta][0][a]['approx_idxs']
                            
                            if float(deadlock_string) > 0:
                                # Absorbing state has index zero
                                subsubstring_a = [substring_start+' 0 '+deadlock_string+' '+actionLabel]
                            else:
                                subsubstring_a = []
                                
                            # Add resulting entries to the list
                            subsubstring_b = [substring_start +" "+str(i+1)+" "+intv+" "+actionLabel 
                                          for (i,intv) in zip(probability_idxs,probability_strings) if intv]
                            
                        # Increase choice counter
                        choice += 1
                        nr_choices_absolute += 1
                        
                        nr_transitions_absolute += len(subsubstring_a) + len(subsubstring_b)
                        
                        subsubstring[a_idx] = '\n'.join(subsubstring_a + subsubstring_b)
                        
                    # subsubstring = '\n'.join(subsubstring)
                    
                else:
                    
                    # No actions enabled, so only add self-loop
                    if selfloop is False:
                        if self.setup.scenarios['switch'] and mode == 'interval':
                            selfloop_prob = '[1.0,1.0]'
                        else:
                            selfloop_prob = '1.0'
                            
                        subsubstring = [str(s+1) +' 0 '+str(s+1)+' '+selfloop_prob]
            
                        selfloop = True
                        
                        # Increment choices and transitions both by one
                        nr_choices_absolute += 1
                        nr_transitions_absolute += 1
                        choice += 1
                    
                    else:
                        subsubstring = []
                        
                substring[delta_idx] = subsubstring
                
            # transition_file_list[s] = '\n'.join(substring)
            transition_file_list[s] = substring
            
        # transition_file_list = '\n'.join(transition_file_list)
        flatten = lambda t: [item for sublist in t for subsublist in sublist for item in subsublist]
        transition_file_list = '\n'.join(flatten(transition_file_list))
        
        print(' ---- String ready; write to file...')
        
        # Header contains nr of states, choices, and transitions
        header = str(self.nr_regions+1)+' '+str(nr_choices_absolute+1)+' '+str(nr_transitions_absolute+1)+'\n'
        
        if self.setup.scenarios['switch'] and mode == 'interval':
            firstrow = '0 0 0 [1.0,1.0]\n'
        else:
            firstrow = '0 0 0 1.0\n'
        
        # Write content to file
        writeFile(PRISM_transitionfile, 'w', header+firstrow+transition_file_list)
            
        ### Write specification file
        specfile, specification = self.writePRISM_specification(mode, horizon='infinite')
        
        return PRISM_allfiles, specfile, specification