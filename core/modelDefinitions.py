# -*- coding: utf-8 -*-


import numpy as np
import scipy.linalg
import sys

import core.preprocessing.define_gears_order as gears        
import core.preprocessing.user_interface as ui
import core.masterClasses as master


class robot(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] =  [-5]
        self.setup['control']['limits']['uMax'] =  [5]
        
        # Partition size
        self.setup['partition']['nrPerDim']  = [21, 21]
        self.setup['partition']['width']     = [1, 1]
        self.setup['partition']['origin']    = [0, 0]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal']           = [[0, 0]]
        self.setup['specification']['critical']       = None
        
        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.15
    
    def setModel(self, tau, observer):
        
        self.tau = tau
        
        # State transition matrix
        self.A  = np.array([[1, tau],
                                [0, 1]])
        
        # Input matrix
        self.B  = np.array([[tau**2/2],
                                [tau]])
        
        if observer:
            # Observation matrix
            self.C          = np.array([[1, 0]])
            self.r          = len(self.C)
        
        # Disturbance matrix
        self.W  = np.array([[0],[0]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*self.setup['noise']['sigma_w_value']
        self.noise['w_mean'] = self.W.flatten()
        

class drone(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        self.modelDim, _  = ui.user_choice('model dimension',[2,3])
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [-6]
        self.setup['control']['limits']['uMax'] = [6]
    
        if self.modelDim == 2:
    
            # Partition size
            self.setup['partition']['nrPerDim']  = [9,9,9,9]#[11, 11, 11, 11]
            self.setup['partition']['width']     = [2, 2, 2, 2]
            self.setup['partition']['origin']    = [0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal']           = [[a,b,c,d] 
                             for a in [6,8] 
                             for b in np.arange(-10,10+1,2) 
                             for c in [6,8] 
                             for d in np.arange(-10,10+1,2)]
            self.setup['specification']['critical']       = [[a,b,c,d] 
                             for a in [-8,-6,-4] 
                             for b in np.arange(-10,10+1,2) 
                             for c in [-2,0] 
                             for d in np.arange(-10,10+1,2)] + \
                                                            [[a,b,c,d]
                             for a in [4,6] 
                             for b in np.arange(-10,10+1,2) 
                             for c in [-8,-6,-4,-2,0,2] 
                             for d in np.arange(-10,10+1,2)]
            
        elif self.modelDim == 3:
            
            # Partition size
            self.setup['partition']['nrPerDim']  = [7, 7, 7, 7, 7, 7]
            self.setup['partition']['width']     = [2, 2, 2, 2, 2, 2]
            self.setup['partition']['origin']    = [0, 0, 0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = [5, 5, 5, 5, 5, 5]
            self.setup['targets']['domain']      = [7,7,7,7,7,7]
            
            # Specification information
            self.setup['specification']['goal']       = [[0, 0, 0, 0, 0, 0]]
            self.setup['specification']['critical']   = None
        
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()

        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.15

    def setModel(self, tau, observer):
        '''
        Define LTI system for the robot defined on a 2-dimensional state space
    
        Parameters
        ----------
        model : dict
            Input model dictionary.
    
        Returns
        -------
        model : dict
            Output model dictionary, with LTI system definitions added to it.
    
        '''
           
        self.tau = tau
        
        # State transition matrix
        Ablock = np.array([[1, tau],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[tau**2/2],
                           [tau]])
        
        if self.modelDim==3:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)
            
            # Disturbance matrix
            self.W  = np.array([[0],[0],[0],[0],[0],[0]])
            
            if observer:
                # Observation matrix
                self.C          = np.array([[1, 0, 1, 0, 1, 0]])
                self.r          = len(self.C)
                
        else:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock)
        
            # Disturbance matrix
            self.W  = np.array([[0],[0],[0],[0]])
        
            if observer:
                # Observation matrix
                self.C          = np.array([[1, 0, 1, 0]])
                self.r          = len(self.C)
            
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*self.setup['noise']['sigma_w_value']
        self.noise['w_mean'] = self.W.flatten()
       
        
class aircraft_altitude(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [-100]
        self.setup['control']['limits']['uMax'] = [100]
    
        # Partition size
        self.setup['partition']['nrPerDim']  = [7, 7, 7, 7,]
        self.setup['partition']['width']     = [2, 2, 2, 2,]
        self.setup['partition']['origin']    = [0, 0, 0, 0]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.setup['targets']['nrPerDim']    = [3,  5,  5,  55]
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal']           = [[0, 0, 0, 0]]
        self.setup['specification']['critical']       = None
        
        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.1

    def setModel(self, tau, observer):
        '''
    
        Parameters
        ----------
        model : dict
            Input model dictionary.
    
        Returns
        -------
        model : dict
            Output model dictionary, with LTI system definitions added to it.
    
        '''
        
        self.tau = tau
        
        # Discretization time
        n = 4
        
        A_cont = np.array([
                [-0.4272e-1,    -0.8541e1,      -0.4451,        -0.3216e2],
                [-0.7881e-3,    -0.5291,        0.9869,         0.1639e-09],
                [0.4010e-3,     0.3541e1,       -0.2228,        0.6150e-8],
                [0,             0,              0.1e1,          0]
            ])
        
        B_cont = np.array([
                [-0.3385e-1,    0.9386e-1,  0.4888e-2],
                [-0.1028e-2,    -0.1297e-2, -0.4054e-3],
                [0.2718e-1,     -0.5744e-2, 0.1351e-1],
                [0,             0,          0]
            ])
        
        W_cont = np.array([
                [0],
                [0],
                [0],
                [0],
            ])
        
        # Discretize model with respect to time
        alpha, beta0, alphaSum  = gears.gears_order(s=1)
        
        A_bar = np.linalg.inv( np.eye(n) - tau*beta0*A_cont )
        O_bar = tau * beta0 * A_bar
        
        self.A = A_bar * alphaSum
        self.B = O_bar @ B_cont
        self.W = O_bar @ W_cont
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*self.setup['noise']['sigma_w_value']
        self.noise['w_mean'] = self.W.flatten()


class anaesthesia_delivery(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [0]
        self.setup['control']['limits']['uMax'] = [100]
    
        # Partition size
        self.setup['partition']['nrPerDim']  = [5, 25, 100]
        self.setup['partition']['width']     = [0.01, 0.01, 0.001]
        self.setup['partition']['origin']    = [0, 0, 0]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal']           = [[0, 0, 0]]
        self.setup['specification']['critical']       = None
        
        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 1e-3

    def setModel(self, tau, observer):
        '''
        Define 3-dimensional linear dynamical system for anaesthesia delivery
        benchmark
    
        Parameters
        ----------
        model : dict
            Input model dictionary.
    
        Returns
        -------
        model : dict
            Output model dictionary, with LTI system definitions added to it.
    
        '''
        
        self.tau = tau
        
        # Discretization time
        n   = 3
        
        k10 = 0.4436
        k12 = 0.1140
        k13 = 0.0419
        k21 = 0.0550
        k31 = 0.0033
        V1  = 16.044
        
        # 2 zone --> 3 state variables
        A_cont = np.array([
        [-(k10 + k12 + k13),    k12,       k13],
        [k21,                  -k21,       0],
        [k31,                   0,         -k31]
        ])
       
        B_cont  = np.array([[1/V1],
                                     [0.2],
                                     [0.2]])
        
        W_cont  = np.array([
                [0],
                [0],
                [0]
                ])
            
        # Discretize model with respect to time
        alpha, beta0, alphaSum  = gears.gears_order(s=1)
        
        A_bar = np.linalg.inv( np.eye(n) - tau*beta0*A_cont )
        O_bar = tau * beta0 * A_bar
        
        self.A = A_bar * alphaSum
        self.B = O_bar @ B_cont
        self.W = O_bar @ W_cont
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*self.setup['noise']['sigma_w_value']
        self.noise['w_mean'] = self.W.flatten()
        
        
class cold_store(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        self.modelDim, _  = ui.user_choice('number of temperature zones',[2,3])
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [0]
        self.setup['control']['limits']['uMax'] = [1500]
    
        if self.modelDim == 2:
    
            # Partition size
            self.setup['partition']['nrPerDim']  = [23, 23, 23]
            self.setup['partition']['width']     = [14/23, 14/23, 14/23]
            self.setup['partition']['origin']    = [-5, -5, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal']           = [[-5, -5, 0]]
            self.setup['specification']['critical']       = None
            
        elif self.modelDim == 3:
            
            # Partition size
            self.setup['partition']['nrPerDim']  = [23, 23, 23, 23]
            self.setup['partition']['width']     = [14/23, 14/23, 14/23, 14/23]
            self.setup['partition']['origin']    = [-5, -5, -5, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal']           = [-5, -5, -5, 0]
            self.setup['specification']['critical']       = None
        
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()

        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.05

    def setModel(self, tau, observer):           
        '''
        Define LTI system for the cold store multiple of temperature zones
    
        Parameters
        ----------
        model : dict
            Input model dictionary.
        zones : int, default=1
            The number of temperature zones in the cold store
    
        Returns
        -------
        model : dict
            Output model dictionary, with LTI system definitions added to it.
    
        '''
            
        # Define number of state features
        n       = self.modelDim + 1
        
        # Discretization time
        tau     = 180
        
        # Ambient temperature
        T_amb   = 10
        
        # Define continuous thermal building model
        Rwall0  = 0.01075189*2
        Cwall   = 235446/4
        
        # Resistance and capacitance of zones (depends on the number of zones)
        Rwall1  = (0.01075189/4) * self.modelDim
        Rwall2  = (0.01075189/4) * self.modelDim
        Rwall3  = (0.01075189/4) * self.modelDim   
        C1      = (50000/2) / self.modelDim
        C2      = (50000/2) / self.modelDim
        C3      = (50000/2) / self.modelDim
        
        # Resistance between different zones
        R12      = 0.01
        R23      = 0.01
        R13      = 0.01
        
        if self.modelDim == 1:
            # 1 zone --> 2 state variables
            A_cont = np.array([
            [-1/C1*(1/Rwall1),       1/(C1*Rwall1)],
            [1/(Cwall*Rwall1),      -1/Cwall*(1/Rwall1 + 1/Rwall0)]            
            ])
           
            B_cont  = np.array([[-1/C1],
                                [0]
                                ])
            
            W_cont  = np.array([
                    [0],
                    [T_amb / (Cwall*Rwall0)]
                    ])
            
            if observer:
                # Observation model
                self.C = np.array([[1,  0]])
            
        if self.modelDim == 2:
            # 2 zone --> 3 state variables
            A_cont = np.array([
            [-1/C1*(1/Rwall1 + 1/R12),            1/(C1*R12),                          1/(C1*Rwall1)],
            [1/(C2*R12),                         -1/C2*(1/Rwall2 + 1/R12),             1/(C2*Rwall2)],
            [1/(Cwall*Rwall1),                    1/(Cwall*Rwall2),                   -1/Cwall*(1/Rwall0 + 1/Rwall1 + 1/Rwall2)]
            ])
           
            B_cont  = np.array([[-1/C1,        0],
                                [0,            -1/C2],
                                [0,            0]])
            
            W_cont  = np.array([
                    [0],
                    [0],
                    [T_amb / (Cwall*Rwall0)]
                    ])
            
            if observer:
                # Observation model
                self.C = np.array([[1,  0,  0]])
            
        if self.modelDim == 3:
            # 3 zone --> 4 state variables   
            A_cont = np.array([
            [-1/C1*(1/Rwall1 + 1/R12 + 1/R13),    1/(C1*R12),                          1/(C1*R13),                          1/(C1*Rwall1)],
            [1/(C2*R12),                         -1/C2*(1/Rwall2 + 1/R12 + 1/R23),     1/(C2*R23),                          1/(C2*Rwall2)],
            [1/(C3*R13),                          1/(C3*R23),                         -1/C3*(1/Rwall3 + 1/R13 + 1/R23),     1/(C3*Rwall3)],
            [1/(Cwall*Rwall1),                    1/(Cwall*Rwall2),                    1/(Cwall*Rwall3),                   -1/Cwall*(1/Rwall0 + 1/Rwall1 + 1/Rwall2 + 1/Rwall3)]
            ])
           
            B_cont  = np.array([[-1/C1,        0,          0],
                                [0,            -1/C2,      0],
                                [0,            0,          -1/C3],
                                [0,            0,          0]])
            
            W_cont  = np.array([
                    [0],
                    [0],
                    [0],
                    [T_amb / (Cwall*Rwall0)]
                    ])
            
            if observer:
                # Observation model
                self.C = np.array([[1,  0,  0,  0]])
        
        # Discretize model with respect to time
        alpha, beta0, alphaSum  = gears.gears_order(s=1)
        
        A_bar = np.linalg.inv( np.eye(n) - tau*beta0*A_cont )
        O_bar = tau * beta0 * A_bar
        
        self.A = A_bar * alphaSum
        self.B = O_bar @ B_cont
        self.W = O_bar @ W_cont
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*self.setup['noise']['sigma_w_value']
        self.noise['w_mean'] = self.W.flatten()