# -*- coding: utf-8 -*-


import numpy as np
import scipy.linalg
import sys

import core.preprocessing.define_gears_order as gears        
import core.preprocessing.user_interface as ui
import core.masterClasses as master

from .commons import setStateBlock

class robot(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] =  [-3]
        self.setup['control']['limits']['uMax'] =  [3]
        
        # Partition size
        self.setup['partition']['nrPerDim']  = [7, 4]
        self.setup['partition']['width']     = [2, 1]
        self.setup['partition']['origin']    = [0, 0]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'

        # Specification information
        self.setup['specification']['goal']           = [[0, 0]]
        self.setup['specification']['critical']       = [[]]
        
        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.15
        
        self.tau = 1.0
    
    def setModel(self, observer):
        
        # State transition matrix
        self.A  = np.array([[1, self.tau],
                                [0, 1]])
        
        # Input matrix
        self.B  = np.array([[self.tau**2/2],
                                [self.tau]])
        
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
        

class UAV(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        self.modelDim, _  = ui.user_choice('model dimension',[2,3])
    
        if self.modelDim == 2:
    
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [-4, -4]
            self.setup['control']['limits']['uMax'] = [4, 4]        
    
            # Partition size
            self.setup['partition']['nrPerDim']  = [9,5,9,5]#[11, 11, 11, 11]
            self.setup['partition']['width']     = [2, 1.5, 2, 1.5]
            self.setup['partition']['origin']    = [0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[-6,-8], b='all', c=[6,8], d='all')
            self.setup['specification']['critical'] = np.vstack((
                setStateBlock(self.setup['partition'], a=[-8,-6,-4], b='all', c=[-2,0], d='all'),
                setStateBlock(self.setup['partition'], a=[4,6], b='all', c=[-8,-6,-4,-2,0,2], d='all')
                ))
            
        elif self.modelDim == 3:
            
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [-4, -4, -4]
            self.setup['control']['limits']['uMax'] = [4, 4, 4]
            
            # Partition size
            self.setup['partition']['nrPerDim']  = [4, 4, 4, 4, 4, 4] #[7, 7, 7, 7, 7, 7]
            self.setup['partition']['width']     = [2, 2, 2, 2, 2, 2] #[2, 2, 2, 2, 2, 2]
            self.setup['partition']['origin']    = [0, 0, 0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto' #[5, 5, 5, 5, 5, 5]
            self.setup['targets']['domain']      = 'auto' #[7,7,7,7,7,7]
            
            # Specification information
            self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[3], b='all', c=[3], d='all', e=[3], f='all')
            
            self.setup['specification']['critical']   = np.vstack((
                setStateBlock(self.setup['partition'], a=[-3,-1], b='all', c=[1,3], d='all', e=[-3,-1], f='all'),
                setStateBlock(self.setup['partition'], a=[-3,-1], b='all', c=[-3,-1], d='all', e=[3], f='all')
                ))
        
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()

        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.15
        
        self.tau = 1
        
    def setModel(self, observer):
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
        
        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
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
        
    def setTurbulenceNoise(self, N):
        
        from core.UAV.dryden import DrydenGustModel
        
        # V_a = speed in 
        turb = DrydenGustModel(dt=1, b=5, h=20, V_a = 25, intensity="moderate")
        
        iters = N
        sample_length = 20
        dim = 3
        
        samples = np.zeros((iters,dim))
        
        for i in range(iters):
            
            if i % 100 == 0:
                print(' -- Create turbulence noise sample:',i)
            
            turb.reset()
            turb.simulate(sample_length)
            timeseries = turb.vel_lin
            
            samples[i,:] = timeseries[:,-1] / 10
            
        self.noise['samples'] = samples
        self.noise['w_mean'] *= 0

        
class UAV_v2(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        self.modelDim, _  = ui.user_choice('model dimension',[2,3])
    
        if self.modelDim == 2:
    
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [-5, -5]
            self.setup['control']['limits']['uMax'] = [5, 5]        
    
            # Partition size
            self.setup['partition']['nrPerDim']  = [7,4,7,4]#[11, 11, 11, 11]
            self.setup['partition']['width']     = [2, 1, 2, 1]
            self.setup['partition']['origin']    = [0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[6], b='all', c=[6], d='all')
            
            self.setup['specification']['critical'] = np.vstack((
                setStateBlock(self.setup['partition'], a=[-6,-4,-2], b='all', c=[2], d='all'),
                setStateBlock(self.setup['partition'], a=[4,6], b='all', c=[-6,-4], d='all')
                ))
            
        elif self.modelDim == 3:
            
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [-4, -4, -4]
            self.setup['control']['limits']['uMax'] = [4, 4, 4]
            
            # Partition size
            self.setup['partition']['nrPerDim']  = [7, 5, 7, 5, 7, 5] #[7, 7, 7, 7, 7, 7]
            self.setup['partition']['width']     = [2, 1.5, 2, 1.5, 2, 1.5] #[2, 2, 2, 2, 2, 2]
            self.setup['partition']['origin']    = [0, 0, 0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[4,6], b='all', c=[4,6], d='all', e=[4,6], f='all')
            
            self.setup['specification']['critical']   = np.vstack((
                setStateBlock(self.setup['partition'], a=[-2,0], b='all', c=[2,4,6], d='all', e='all', f='all'),
                setStateBlock(self.setup['partition'], a=[-6,-4], b='all', c=[4,6], d='all', e=[4,6], f='all'),
                setStateBlock(self.setup['partition'], a=[-2,0], b='all', c=[-6,-4,-2,0], d='all', e=[-6,-4], f='all'),
                setStateBlock(self.setup['partition'], a=[2,4,6], b='all', c=[-6,-4], d='all', e=[-6,-4], f='all')
                ))
        
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()

        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.15
        
        self.tau = 1

    def setModel(self, observer):
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
        
        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
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
   
    def setTurbulenceNoise(self, N):
        
        from core.UAV.dryden import DrydenGustModel
        
        # V_a = speed in 
        turb = DrydenGustModel(dt=1, b=5, h=20, V_a = 25, intensity="moderate")
        
        iters = N
        sample_length = 20
        dim = 3
        
        samples = np.zeros((iters,dim))
        
        for i in range(iters):
            
            if i % 100 == 0:
                print(' -- Create turbulence noise sample:',i)
            
            turb.reset()
            turb.simulate(sample_length)
            timeseries = turb.vel_lin
            
            samples[i,:] = timeseries[:,-1] / 10
            
        self.noise['samples'] = samples
        self.noise['w_mean'] *= 0
   
   
class building_2room(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [11, -5] #[15, -5]
        self.setup['control']['limits']['uMax'] = [31, 5] #[25, 5]
            
        # Partition size
        self.setup['partition']['nrPerDim']  = [21,21,9,9]#[21,     21,     11,     11]
        self.setup['partition']['width']     = [0.2,   0.2,   1,      1]
        self.setup['partition']['origin']    = [21,     21,     35,     35]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[21], b=[21], c='all', d='all')
        
        self.setup['specification']['critical'] = [[]]
        
        self.tau = 15 # NOTE: in minutes for BAS!

    def setModel(self, observer):           
        '''
    
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
        
        import core.BAS.parameters as BAS_class
        
        BAS = BAS_class.parameters()
        
        # Steady state values
        Tswb    = BAS.Boiler['Tswbss']         
        Twss    = BAS.Zone1['Twss']
        Trwrss  = BAS.Radiator['Trwrss']
        Pout1   = BAS.Radiator['Zone1']['Prad']    
        Pout2   = 2*BAS.Radiator['Zone2']['Prad']    
        
        w       = BAS.Radiator['w_r']
        
        BAS.Zone1['Cz'] = BAS.Zone1['Cz'] / 5
        
        m1      = BAS.Zone1['m'] * 2
        m2      = BAS.Zone2['m']
        
        k1_a    = 3*BAS.Radiator['k1']
        k1_b    = BAS.Radiator['k1']
        
        k0_a    = 3*BAS.Radiator['k0']
        k0_b    = BAS.Radiator['k0']
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((4,4));
        A_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
        A_cont[0,2] = (Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
        A_cont[1,1] = -(1/(BAS.Zone2['Rn']*BAS.Zone2['Cz']))-(Pout2*BAS.Radiator['alpha2'] )/(BAS.Zone2['Cz']) - (m2*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz'])
        A_cont[1,3] = (Pout2*BAS.Radiator['alpha2'] )/(BAS.Zone2['Cz'])
        A_cont[2,0] = (k1_a)
        A_cont[2,2] = -(k0_a*w) - k1_a
        A_cont[3,1] = (k1_b)
        A_cont[3,3] = -(k0_b*w) - k1_b
        
        # B_cont      = np.zeros((4,2))
        # B_cont[0,0] = (m*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        # B_cont[1,1] = (m*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz'])
        # B_cont[2,0] = 0
        # B_cont[3,1] =0

        B_cont      = np.zeros((4,2))
        B_cont[0,0] = (m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        B_cont[1,0] = (m2*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz'])
        B_cont[2,1] = (k0_a*w*Tswb) # < Allows to change the boiler temperature
        B_cont[3,1] = (k0_b*w*Tswb) # < Allows to change the boiler temperature

        # B_cont = np.array([
        #         [ (m*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']) ],
        #         [ (m*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz']) ],
        #         [ 0 ],
        #         [ 0 ]
        #         ])
        
        W_cont  = np.array([
                [ (Twss/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))+ (BAS.Radiator['alpha1'])/(BAS.Zone1['Cz']) ],
                [ ((Twss-1)/(BAS.Zone2['Rn']*BAS.Zone2['Cz']))+ (BAS.Radiator['alpha1'])/(BAS.Zone1['Cz']) ],
                [ (k0_a*w*Tswb) ],
                [ (k0_b*w*Tswb) ]
                ])
        
        # Discretize model with respect to time
        # self.A, self.B, self.W = discretizeGearsMethod(A_cont, B_cont, W_cont, self.tau)
        
        self.A = np.eye(4) + self.tau*A_cont
        self.B = B_cont*self.tau
        self.W = W_cont*self.tau
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = 0.2*np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Zone2['Tz']['sigma'], BAS.Radiator['rw']['sigma'], BAS.Radiator['rw']['sigma'] ])
        
        self.noise['w_mean'] = self.W.flatten()
        
        
class building_1room(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [15, -5]
        self.setup['control']['limits']['uMax'] = [25, 5]
            
        # Partition size
        self.setup['partition']['nrPerDim']  = [21, 11]
        self.setup['partition']['width']     = [0.25, 1]
        self.setup['partition']['origin']    = [21, 32]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], a=[21], b='all')
        
        self.setup['specification']['critical'] = [[]]
        
        self.tau = 15 # NOTE: in minutes for BAS!

    def setModel(self, observer):           
        '''
    
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
        
        import core.BAS.parameters as BAS_class
        
        BAS = BAS_class.parameters()
        
        # Steady state values
        Tswb    = BAS.Boiler['Tswbss']
        Tsp     = BAS.Zone1['Tsp']             
        Twss    = BAS.Zone1['Twss']
        Trwrss  = BAS.Radiator['Trwrss']
        Pout1   = BAS.Radiator['Zone1']['Prad']    
        Pout2   = BAS.Radiator['Zone2']['Prad']    
        m       = 1.5*BAS.Zone1['m']
        w       = BAS.Radiator['w_r']
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((2,2));
        A_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
        A_cont[0,1] = (Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
        A_cont[1,0] = (BAS.Radiator['k1'])
        A_cont[1,1] = -(BAS.Radiator['k0']*w) - BAS.Radiator['k1']
        
        B_cont      = np.zeros((2,2))
        B_cont[0,0] = (m*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        B_cont[1,1] = (BAS.Radiator['k0']*w*Tswb) # < Allows to change the boiler temperature

        
        W_cont  = np.array([
                [ (Twss/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))+ (BAS.Radiator['alpha1'])/(BAS.Zone1['Cz']) ],
                [ (BAS.Radiator['k0']*w*Tswb) ],
                ])
        
        # Discretize model with respect to time
        # self.A, self.B, self.W = discretizeGearsMethod(A_cont, B_cont, W_cont, self.tau)
        
        self.A = np.eye(2) + self.tau*A_cont
        self.B = B_cont*self.tau
        self.W = W_cont*self.tau
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = 0.5*np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ])
        
        self.noise['w_mean'] = self.W.flatten()
        
#####################################
# BELOW ARE VARIOUS (OUTDATED) MODELS
#####################################
   
        
class aircraft_altitude(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [-100, -100, -100]
        self.setup['control']['limits']['uMax'] = [100, 100, 100]
    
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
        
        self.tau = 1

    def setModel(self, observer):
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
        self.A, self.B, self.W = discretizeGearsMethod(A_cont, B_cont, W_cont, self.tau)
        
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
        
        self.tau = 30

    def setModel(self, observer):
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
        self.A, self.B, self.W = discretizeGearsMethod(A_cont, B_cont, W_cont, self.tau)
        
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
    
        if self.modelDim == 2:
    
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [0, 0]
            self.setup['control']['limits']['uMax'] = [1500, 1500]        
    
            # Partition size
            self.setup['partition']['nrPerDim']  = [23, 23, 23]
            self.setup['partition']['width']     = [14/23, 14/23, 14/23]
            self.setup['partition']['origin']    = [-5, -5, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal']     = [[-5, -5, 0]]
            self.setup['specification']['critical'] = None
            
        elif self.modelDim == 3:
            
            # Authority limit for the control u, both positive and negative
            self.setup['control']['limits']['uMin'] = [0, 0 ,0]
            self.setup['control']['limits']['uMax'] = [1500, 1500, 1500]
            
            # Partition size
            self.setup['partition']['nrPerDim']  = [23, 23, 23, 23]
            self.setup['partition']['width']     = [14/23, 14/23, 14/23, 14/23]
            self.setup['partition']['origin']    = [-5, -5, -5, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.setup['targets']['nrPerDim']    = 'auto'
            self.setup['targets']['domain']      = 'auto'
            
            # Specification information
            self.setup['specification']['goal']     = [[-5, -5, -5, 0]]
            self.setup['specification']['critical'] = None
        
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()

        # Covariance values of the process noise (w) and measurement noise (v)
        self.setup['noise']['sigma_w_value'] = 0.05
        
        self.tau = 180

    def setModel(self, observer):           
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
        self.A, self.B, self.W = discretizeGearsMethod(A_cont, B_cont, W_cont, self.tau)
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*self.setup['noise']['sigma_w_value']
        self.noise['w_mean'] = self.W.flatten()
        

        
        
def discretizeGearsMethod(A_cont, B_cont, W_cont, tau):
    
    # Dimension of the state
    n = len(A_cont)
    
    # Discretize model with respect to time
    alpha, beta0, alphaSum  = gears.gears_order(s=1)
    
    A_bar = np.linalg.inv( np.eye(n) - tau*beta0*A_cont )
    O_bar = tau * beta0 * A_bar
    
    A = A_bar * alphaSum
    B = O_bar @ B_cont
    W = O_bar @ W_cont
    
    return A,B,W