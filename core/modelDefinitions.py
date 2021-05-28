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
        self.setup['control']['limits']['uMin'] =  [-5]
        self.setup['control']['limits']['uMax'] =  [8]
        
        # Partition size
        self.setup['partition']['nrPerDim']  = [21, 21]
        self.setup['partition']['width']     = [2, 2]
        self.setup['partition']['origin']    = [-10, 5]
        
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
        self.W  = np.array([[3.5],[-0.7]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)
        
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A,1))*self.setup['noise']['sigma_w_value']

        
class UAV(master.LTI_master):
    
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
                setStateBlock(self.setup['partition'], a=[2,4,6], b='all', c=[-6,-4], d='all', e=[-6,-4], f='all'),
                setStateBlock(self.setup['partition'], a=[-2,0], b='all', c=[-6,-4], d='all', e=[-2,0], f='all')
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
   
    def setTurbulenceNoise(self, N):
        
        samples = np.genfromtxt('input/TurbulenceNoise_N=1000_dim=3.csv', delimiter=',')
            
        self.noise['samples'] = samples
        
class building_2room(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        import core.BAS.parameters as BAS_class
        
        self.BAS = BAS_class.parameters()
        
        # Let the user make a choice for the model dimension
        
        T_boiler = self.BAS.Boiler['Tswbss']
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [14, 14, T_boiler-10, T_boiler-10]
        self.setup['control']['limits']['uMax'] = [26, 26, T_boiler+10, T_boiler+10]
            
        # Partition size
        self.setup['partition']['nrPerDim']  = [21,21,9,9] #11,11] #29,29]
        self.setup['partition']['width']     = [0.2, 0.2, 0.2, 0.2]
        self.setup['partition']['origin']    = [20, 20, 38.3, 38.3] #[19, 19 , 41.57227145, 41.55259975] #[21, 21, 41, 41]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.setup['targets']['nrPerDim']    = 'auto'
        self.setup['targets']['domain']      = 'auto'
        
        # Specification information
        self.setup['specification']['goal'] = setStateBlock(self.setup['partition'], 
                                    a=[20], 
                                    b=[20], 
                                    c='all', d='all') #setStateBlock(self.setup['partition'], a=[21], b=[21], c='all', d='all')
        
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
        
        BAS = self.BAS
        
        # Steady state values
        Twss        = BAS.Zone1['Twss'] + 5
        Pout1       = BAS.Radiator['Zone1']['Prad'] * 1.5
        Pout2       = BAS.Radiator['Zone2']['Prad'] * 1.5
        
        w1          = BAS.Radiator['w_r'] * 1.5
        w2          = BAS.Radiator['w_r'] * 1.5
        
        BAS.Zone1['Cz'] = BAS.Zone1['Cz']
        BAS.Zone1['Rn'] = BAS.Zone1['Rn']
        
        BAS.Zone2['Cz'] = BAS.Zone2['Cz']
        BAS.Zone2['Rn'] = BAS.Zone2['Rn']
        
        m1          = BAS.Zone1['m']
        m2          = BAS.Zone2['m']
        
        Rad_k1_z1   = BAS.Radiator['k1'] * 5
        Rad_k1_z2   = BAS.Radiator['k1'] * 5
        
        Rad_k0_z1   = BAS.Radiator['k0']
        Rad_k0_z2   = BAS.Radiator['k0']
        
        alpha1_z1   = BAS.Radiator['alpha1']
        alpha1_z2   = BAS.Radiator['alpha1']
        
        alpha2_z1   = BAS.Radiator['alpha1']
        alpha2_z2   = BAS.Radiator['alpha1']
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((4,4));
        
        # Room 1
        A_cont[0,0] = ( -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((Pout1*alpha2_z1 )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])) - (1/(BAS.Zone1['Rn']*BAS.Zone1['Cz'])) )
        A_cont[0,2] = (Pout1*alpha2_z1 )/(BAS.Zone1['Cz'])
        
        # Room 2
        A_cont[1,1] = ( -(1/(BAS.Zone2['Rn']*BAS.Zone2['Cz']))-((Pout2*alpha2_z2 )/(BAS.Zone2['Cz'])) - ((m2*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz'])) - (1/(BAS.Zone2['Rn']*BAS.Zone2['Cz'])) )
        A_cont[1,3] = (Pout2*alpha2_z2 )/(BAS.Zone2['Cz'])
        
        # Heat transfer room 1 <-> room 2
        A_cont[0,1] = ( (1/(BAS.Zone1['Rn']*BAS.Zone1['Cz'])) )
        A_cont[1,0] = ( (1/(BAS.Zone2['Rn']*BAS.Zone2['Cz'])) )
        
        # Radiator 1
        A_cont[2,0] = (Rad_k1_z1)
        A_cont[2,2] = ( -(Rad_k0_z1*w1) - Rad_k1_z1 )
        
        # Radiator 2
        A_cont[3,1] = (Rad_k1_z2)
        A_cont[3,3] = ( -(Rad_k0_z2*w2) - Rad_k1_z2 )

        B_cont      = np.zeros((4,4))
        B_cont[0,0] = (m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        B_cont[1,1] = (m2*BAS.Materials['air']['Cpa'])/(BAS.Zone2['Cz'])
        B_cont[2,2] = (Rad_k0_z1*w1) # < Allows to change the boiler temperature
        B_cont[3,3] = (Rad_k0_z2*w2) # < Allows to change the boiler temperature

        W_cont  = np.array([
                [ ((Twss)/(BAS.Zone1['Rn']*BAS.Zone1['Cz'])) + (alpha1_z1)/(BAS.Zone1['Cz']) ],
                [ ((Twss-2)/(BAS.Zone2['Rn']*BAS.Zone2['Cz'])) + (alpha1_z2)/(BAS.Zone1['Cz']) ],
                [ 0 ],
                [ 0 ]
                ])
        
        # Discretize model with respect to time
        # self.A, self.B, self.W = discretizeGearsMethod(A_cont, B_cont, W_cont, self.tau)
        
        self.A = np.eye(4) + self.tau*A_cont
        self.B = B_cont*self.tau
        self.W = W_cont*self.tau
        
        #######
        
        # self.A = np.array([[0.6682, 0, 0.02632, 0],[0, 0.6830, 0, 0.02096],[1.0005, 0, -0.000499, 0],[0, 0.8004, 0, 0.1996]])
        # self.B = np.diag([0.1320,0.1402,self.tau**Rad_k0_z1*w1,self.tau*Rad_k0_z2*w2])
        # self.W = np.array([[3.4364],[2.9272],[13.0207],[10.4166]])
        
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.noise = dict()
        self.noise['w_cov'] = 0.05*np.diag([0.2, 0.2, 0.2, 0.2])
        
        #0.1*np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Zone2['Tz']['sigma'], BAS.Radiator['rw']['sigma'], BAS.Radiator['rw']['sigma'] ])
        
        self.A_cont = A_cont
        self.B_cont = B_cont
        self.W_cont = W_cont
        
class building_1room(master.LTI_master):
    
    def __init__(self):
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Let the user make a choice for the model dimension
        
        # Authority limit for the control u, both positive and negative
        self.setup['control']['limits']['uMin'] = [14, -10]
        self.setup['control']['limits']['uMax'] = [28, 10]
            
        # Partition size
        self.setup['partition']['nrPerDim']  = [21, 21] #[41, 41]
        self.setup['partition']['width']     = [0.2, 0.2] #[0.1, 0.1]
        self.setup['partition']['origin']    = [21, 38] #[21, 45]
        
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
        Tswb    = BAS.Boiler['Tswbss'] - 20
        Twss    = BAS.Zone1['Twss'] - 2
        Pout1   = BAS.Radiator['Zone1']['Prad']      
        
        w       = BAS.Radiator['w_r']
        
        BAS.Zone1['Cz'] = BAS.Zone1['Cz']
        
        m1      = BAS.Zone1['m'] # Proportional factor for the air conditioning
        
        k1_a    = BAS.Radiator['k1']
        k0_a    = BAS.Radiator['k0'] #Proportional factor for the boiler temp. on radiator temp.
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((2,2));
        A_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
        A_cont[0,1] = (Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
        A_cont[1,0] = (k1_a)
        A_cont[1,1] = -(k0_a*w) - k1_a
        
        B_cont      = np.zeros((2,2))
        B_cont[0,0] = (m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        B_cont[1,1] = (k0_a*w) # < Allows to change the boiler temperature

        
        W_cont  = np.array([
                [ (Twss/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))+ (BAS.Radiator['alpha1'])/(BAS.Zone1['Cz']) ],
                [ (k0_a*w*Tswb) ],
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
        self.noise['w_cov'] = np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ])

        
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