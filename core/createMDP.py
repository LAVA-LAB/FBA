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
        self.nr_states  = self.nr_regions * self.N
        self.nr_actions = abstr['nr_actions']
        
        # Specify goal state set
        self.goodStates = [g + x*self.nr_regions for x in range(N) for g in abstr['goal']]
        
        # Specify sets that can never reach target set
        self.badStates  = [c + x*self.nr_regions for x in range(N) for c in abstr['critical']]
        
    def createPython(self, abstr, kalman=False):
        
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
                            
                            to_probs_all    = abstr['prob'][delta][k][action]['approx']
                            
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

    def writePRISM_scenario(self, abstr, mode='default', horizon='infinite'):
        '''
        Converts the model to the PRISM language, and write the model to a file.
    
        Parameters
        ----------
        mode : str, optional
            Is either 'default' or 'interval'.
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
                "module iMDP \n\n",
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
                                
                                # Create scenario-based interval MDP
                                succProbs_lb = floor_decimal(abstr['prob'][delta][k][a]['lb'], 5)
                                succProbs_ub = floor_decimal(abstr['prob'][delta][k][a]['ub'], 5)
                                
                                # If mode is interval, use intervals on probs.
                                succPieces = ["["+
                                              str(floor_decimal(max(1e-4, lb),5))+","+
                                              str(floor_decimal(min(1,    ub),5))+"] : (x'="+
                                              str(i)+")"+kprime 
                                              for i,(lb, ub) in enumerate(zip(succProbs_lb, succProbs_ub)) if ub > 0]
                                
                                # Use absorbing state to make sure that probs sum to one
                                deadlock_lb = floor_decimal(abstr['prob'][delta][k][a]['deadlock_lb'], 5)
                                deadlock_ub = floor_decimal(abstr['prob'][delta][k][a]['deadlock_ub'], 5)
                                succPieces += ["["+
                                              str(floor_decimal(max(1e-4, deadlock_lb),5))+","+
                                              str(floor_decimal(min(1,    deadlock_ub),5))+"] : (x'=-1)"+kprime]
                                
                            else:
                                
                                # Write resulting states with their probabilities
                                succProbs = floor_decimal(abstr['prob'][delta][k][a]['approx'], 5)
                                
                                if mode == 'default':
                                    # If mode is default, use concrete probabilities
                                    succPieces = [str(p)+":(x'="+str(i)+")"+kprime
                                                  for i,p in enumerate(succProbs) if p > 0]
                                    
                                    # Use absorbing state to make sure that probs sum to one
                                    deadlockProb = np.round(1-sum(succProbs),6)
                                    succPieces += [str(deadlockProb)+":(x'=-1)"+kprime]
                                    
                                else:
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
        
        if horizon == 'infinite':
            # Infer number of time steps in horizon (at minimum delta value)
            horizonLen = int(self.N/min(self.setup.deltas))
            
            if mode == 'default':
                # If mode is default, set maximum probability as specification
                specification = 'Pmax=? [ F<='+str(horizonLen)+' "reached" ]'
                
            elif mode == 'interval':
                # If mode is interval, set lower bound of maximum prob. as spec.
                specification = 'Pmaxmin=? [ F<='+str(horizonLen)+' "reached" ]'
            
        else:
            if mode == 'default':
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
        
        # # Define terminal command (may be user-dependent)
        # commandfile = self.setup.directories['outputFcase']+ \
        #     self.setup.mdp['filename']+"_"+mode+"_command.txt"
            
        # # Write command file
        # command = "bin/prism models/m2/Abstraction_interval.prism -pf 'Pmaxmin=?[F<="+\
        #     str(horizonLen)+" \"reached\"]' -exportadv strat.csv -exportstates states.csv -exportvector vector.csv"
        # writeFile(commandfile, 'w', command)
        
        if mode == 'default':
            printSuccess('MDP ('+horizon+' horizon) exported as PRISM file')
        else:
            printSuccess('iMDP ('+horizon+' horizon) exported as PRISM file')
        
        return PRISM_file, specification