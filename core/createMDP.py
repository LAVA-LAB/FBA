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
        self.nr_states  = self.nr_regions
        
        # Specify goal state set
        # self.goodStates = abstr['goal']
        
        # Specify sets that can never reach target set
        # self.badStates  = abstr['critical']

    def writePRISM_scenario(self, abstr, trans, mode='estimate', 
                            horizon='infinite'):
        '''
        Converts the model to the PRISM language, and write the model to file.
    
        Parameters
        ----------
        abstr : dict
            Abstraction dictionary
        trans : dict
            Dictionary of transition probabilities        
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
                        enabledIn = np.setdiff1d(abstr['actions_inv'][delta][a],
                                                 np.concatenate((abstr['goal'][k],abstr['critical'][k])))
                        
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
                            
                            if mode == 'interval':
                                
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
        
        labelPieces = ["(x="+str(x)+") & (k="+str(k)+")" for x in abstr['goal'][k] for k in range(self.N)]
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
        
        if mode == 'estimate':
            printSuccess('MDP ('+horizon+' horizon) exported as PRISM file')
        else:
            printSuccess('iMDP ('+horizon+' horizon) exported as PRISM file')
        
        return PRISM_file, specfile, specification
    
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
            self.horizonLen = int(self.N/min(self.setup.deltas))
        elif horizon == 'steadystate':
            # Infer number of time steps in horizon (at minimum delta value)
            self.horizonLen = int(self.N/min(self.setup.deltas))
        else:
            # We always need to provide a maximum nr. of steps, so make equal
            # to time horizon (even though horizon is also in the state space.)
            self.horizonLen = int(self.N/min(self.setup.deltas))
            
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
                nr_states = self.nr_regions * (self.setup.mdp['k_steady_state'] + 1)
                k_steps = np.arange(self.setup.mdp['k_steady_state'] + 1)
                
            else:
                horizon = 'finite'
                nr_states = self.nr_regions * self.N
                k_steps = np.arange(self.N)
        else:
            horizon = 'infinite'
            nr_states = self.nr_regions
            k_steps = [0]
        
        # Define PRISM filename
        PRISM_allfiles = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".all"
        
        print(' --- Writing PRISM states file')
        
        ### Write states file
        PRISM_statefile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".sta"
        
        state_file_string = '\n'.join(['(x)\n0:(-3)\n1:(-2)\n2:(-1)'] + [str(i+3)+':('+str(i)+')' for i in range(nr_states)])
        
        # Write content to file
        writeFile(PRISM_statefile, 'w', state_file_string)
        
        print(' --- Writing PRISM label file')
        
        ### Write label file
        PRISM_labelfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".lab"
            
        label_file_list = ['0="init" 1="deadlock" 2="reached"'] + \
                          ['0: 1'] + \
                          ['1: 2'] + \
                          ['2: 1'] + \
                          ['' for i in range(nr_states)]
        
        for i in range(self.nr_regions):
            substring = ''
            
            # Check if region is a deadlock state
            for delta in self.setup.deltas:
                if len(abstr['actions'][delta][i]) == 0:
                    substring += ' 1'
                    break
            
            for k in k_steps:
                add = k*self.nr_regions
                label_file_list[i+4 + add] = str(i+3 + add)+': 0' + substring
            
        label_file_string = '\n'.join(label_file_list)
           
        # Write content to file
        writeFile(PRISM_labelfile, 'w', label_file_string)
        
        print(' --- Writing PRISM transition file')
        
        ### Write transition file
        PRISM_transitionfile = self.setup.directories['outputFcase']+ \
            self.setup.mdp['filename']+"_"+mode+".tra"
            
        transition_file_list = ['' for i in k_steps]
            
        nr_choices_absolute = 0
        nr_transitions_absolute = 0
        
        printEvery = min(100, max(1, int(self.nr_regions/10)))
        
        # For every time step
        for k in k_steps:
        
            string = ['' for i in range(self.nr_regions)]
            
            add = k*self.nr_regions
            
            # For every state
            for s in range(self.nr_regions):
                
                
                if s % printEvery == 0:
                    print(' ---- Write for region',s)
                
                # if s in abstr['goal'][k]:
                #     continue
                # elif s in abstr['critical'][k]:
                #     continue
                
                substring = ['' for i in range(len(self.setup.deltas)+1)]
                
                choice = 0
                selfloop = False
                
                deltaNoActions = np.zeros(len(self.setup.deltas))
                
                # Check if region is a deadlock state
                for delta_idx,delta in enumerate(self.setup.deltas):
                    
                    if km == None:
                        # Waiting time is not applicable
                        gamma = 0
                    else:
                        # Retreive waiting time
                        gamma = km[delta]['waiting_time'][k]
                    
                    if horizon == 'infinite':
                        add_succ = add
                    else:
                        add_succ = add + (delta+gamma)*self.nr_regions
                        
                        # Make sure that it is below the max. number of states
                        while add_succ >= nr_states:
                            add_succ -= self.nr_regions
                
                    if len(abstr['actions'][delta][s]) > 0 and (k <= self.N - delta or horizon=='steadystate'):
                
                        subsubstring = ['' for i in range(len(abstr['actions'][delta][s]))]
                
                        # For every enabled action                
                        for a_idx,a in enumerate(abstr['actions'][delta][s]):
                            
                            # Define name of action
                            actionLabel = "a_"+str(a)+"_d_"+str(delta)
                            
                            substring_start = str(s+3+add) +' '+ str(choice)
                            
                            if mode == 'interval':
                            
                                # Add probability to end in absorbing state
                                deadlock_string = trans['prob'][delta][k][a]['deadlock_interval_string']
                                
                                # Add probability to reach the goal
                                goal_string = trans['prob'][delta][k][a]['goal_interval_string']
                                
                                # Add probability to reach the goal
                                critical_string = trans['prob'][delta][k][a]['critical_interval_string']
                            
                                # Retreive probability intervals (and corresponding state ID's)
                                probability_strings = trans['prob'][delta][k][a]['interval_strings']
                                probability_idxs    = trans['prob'][delta][k][a]['interval_idxs']
                            
                                # Absorbing state has index zero
                                subsubstring_a = [substring_start+' 0 '+deadlock_string+' '+actionLabel]
                                
                                if goal_string != None:
                                    subsubstring_a += [substring_start+' 1 '+goal_string+' '+actionLabel]
                                if critical_string != None:
                                    subsubstring_a += [substring_start+' 2 '+critical_string+' '+actionLabel]
                                
                                # Add resulting entries to the list
                                subsubstring_b = [substring_start +" "+str(i+3+add_succ)+" "+intv+" "+actionLabel 
                                              for (i,intv) in zip(probability_idxs,probability_strings) if intv]
                                
                            else:
                                
                                # Add probability to end in absorbing state
                                deadlock_string = str(trans['prob'][delta][k][a]['deadlock_approx'])
                                
                                # Add probability to reach the goal
                                goal_string = str(trans['prob'][delta][k][a]['goal_approx'])
                                
                                # Add probability to reach the goal
                                critical_string = str(trans['prob'][delta][k][a]['goal_approx'])
                                
                                # Retreive probability intervals (and corresponding state ID's)
                                probability_strings = trans['prob'][delta][k][a]['approx_strings']
                                probability_idxs    = trans['prob'][delta][k][a]['approx_idxs']
                                
                                if float(deadlock_string) > 0:
                                    # Absorbing state has index zero
                                    subsubstring_a = [substring_start+' 0 '+deadlock_string+' '+actionLabel]
                                else:
                                    subsubstring_a = []
                                    
                                if float(goal_string) > 0:
                                    # Goal state has index 1
                                    subsubstring_a += [substring_start+' 1 '+goal_string+' '+actionLabel]
                                    
                                if float(critical_string) > 0:
                                    # Critical state has index 2
                                    subsubstring_a += [substring_start+' 2 '+critical_string+' '+actionLabel]
                                    
                                # Add resulting entries to the list
                                subsubstring_b = [substring_start +" "+str(i+3+add_succ)+" "+intv+" "+actionLabel 
                                              for (i,intv) in zip(probability_idxs,probability_strings) if intv]
                                
                            # Increase choice counter
                            choice += 1
                            nr_choices_absolute += 1
                            
                            nr_transitions_absolute += len(subsubstring_a) + len(subsubstring_b)
                            
                            subsubstring[a_idx] = '\n'.join(subsubstring_a + subsubstring_b)
                        
                    else:
                        
                        deltaNoActions[ delta_idx ] = True
                        subsubstring = []
                            
                    substring[delta_idx] = subsubstring
                    
                if all(deltaNoActions == True):
                
                    # No actions enabled, so only add self-loop
                    if selfloop is False:
                        if mode == 'interval':
                            selfloop_prob = '[1.0,1.0]'
                        else:
                            selfloop_prob = '1.0'
                            
                        if k > self.N - min(self.setup.deltas):
                            # If time horizon reached, to absorbing state
                            subsubstring = [str(s+3+add) +' 0 0 '+selfloop_prob]
                        else:
                            # If not horizon reached, self-loop
                            subsubstring = [str(s+3+add) +' 0 '+str(s+3+add)+' '+selfloop_prob]
            
                        selfloop = True
                        
                        # Increment choices and transitions both by one
                        nr_choices_absolute += 1
                        nr_transitions_absolute += 1
                        choice += 1
                
                else:
                    subsubstring = []
                    
                substring[-1] = subsubstring
                    
                string[s] = substring
                    
            transition_file_list[k] = string
            
        # transition_file_list = '\n'.join(transition_file_list)
        flatten = lambda t: [item for sublist in t for subsublist in sublist for subsubsublist in subsublist for item in subsubsublist]
        transition_file_list = '\n'.join(flatten(transition_file_list))
        
        print(' ---- String ready; write to file...')
        
        # Header contains nr of states, choices, and transitions
        size_states = nr_states+3
        size_choices = nr_choices_absolute+3
        size_transitions = nr_transitions_absolute+3
        model_size = {'States': size_states, 
                      'Choices': size_choices, 
                      'Transitions':size_transitions}
        header = str(size_states)+' '+str(size_choices)+' '+str(size_transitions)+'\n'
        
        if mode == 'interval':
            firstrow = '0 0 0 [1.0,1.0]\n1 0 1 [1.0,1.0]\n2 0 2 [1.0,1.0]\n'
        else:
            firstrow = '0 0 0 1.0\n1 0 1 1.0\n2 0 2 1.0\n'
        
        # Write content to file
        writeFile(PRISM_transitionfile, 'w', header+firstrow+transition_file_list)
            
        ### Write specification file
        specfile, specification = self.writePRISM_specification(mode, horizon=horizon)
        
        return model_size, PRISM_allfiles, specfile, specification