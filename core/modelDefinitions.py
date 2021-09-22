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
import scipy.linalg             # Import to enable matrix operations
import sys                      # Import to allow terminating the script

from .preprocessing.define_gears_order import discretizeGearsMethod        
import core.preprocessing.user_interface as ui
import core.masterClasses as master
from .commons import setStateBlock, defSpecBlock

class double_integrator(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize double_integrator class, which is a 1D dummy problem

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.base_delta = 2
        
        rates, _  = ui.user_choice('adaptive measurement rate', \
                                   [[],[2],[2,4],[2,4,6]])
        
        self.adaptive = {'rates': rates,
                         'target_points': np.array([[i, j] for i in range(-20,21,4) for j in range(-8,10,8)])
                         }
        
        # Authority limit for the control u, both positive and negative
        self.control['limits']['uMin'] =  [-5]
        self.control['limits']['uMax'] =  [5]
        
        # Partition size
        self.partition['nrPerDim']  = [21, 21]
        self.partition['width']     = [2, 2] 
        self.partition['origin']    = [0, 0]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.targets['nrPerDim']    = 'auto'
        self.targets['domain']      = 'auto'

        # Specification information
        self.spec['goal']     = {1: defSpecBlock(self.partition, a=[-3,3], b=[-3,3])}
        self.spec['critical'] = {}#{1: defSpecBlock(self.partition, a=[9,11], b=None)}
        
        # Step-bound on property
        self.endTime = 32#64
    
    def setModel(self, observer):
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
        self.LTI = {}
        
        # Discretization step size
        self.LTI['tau'] = 1.0
        
        # State transition matrix
        self.LTI['A']  = np.array([[1, self.LTI['tau']],
                                [0, 1]])
        
        # Input matrix
        self.LTI['B']  = np.array([[self.LTI['tau']**2/2],
                                [self.LTI['tau']]])
        
        self.LTI['noise'] = dict()
        
        if observer:
            # Observation matrix
            self.LTI['C'] = np.array([[1, 0]])
            self.LTI['r'] = len(self.LTI['C'])
            
            self.LTI['noise']['v_cov'] = np.eye(np.size(self.LTI['C'],0))*0.25
            
            self.filter = {'cov0': np.diag([2, 2])}
        
        # Disturbance matrix
        self.LTI['Q']  = np.array([[0],[0]])
        
        # Determine system dimensions
        self.LTI['n'] = np.size(self.LTI['A'],1)
        self.LTI['p'] = np.size(self.LTI['B'],1)
        
        # Covariance of the process noise
        self.LTI['noise']['w_cov'] = np.eye(np.size(self.LTI['A'],1))*0.25

        
class UAV(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the UAV model class, which can be 2D or 3D. The 3D case
        corresponds to the UAV benchmark in the paper.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        #self.deltas = [2,3]#,4]
        self.base_delta = 2
        
        # Let the user make a choice for the model dimension
        self.modelDim, _  = ui.user_choice('model dimension',[2,3])
    
        if self.modelDim == 2:
            
            self.adaptive = {'rates': [2],
                #'target_points': np.array([[i, 0, j, 0] for i in range(-8,10,4) for j in range(-8,10,4)])
                'target_points': np.array([[i, x, j, y] for i in [6,8] for j in [-4,-2] for x in [-1.5, 0, 1.5] for y in [-1.5, 0, 1.5]])
                }
    
            # Authority limit for the control u, both positive and negative
            self.control['limits']['uMin'] = [-4, -4]
            self.control['limits']['uMax'] = [4, 4]        
    
            # Partition size
            self.partition['nrPerDim']  = [11,5,11,5]
            self.partition['width']     = [2, 1.5, 2, 1.5]
            self.partition['origin']    = [0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.targets['nrPerDim']    = 'auto'
            self.targets['domain']      = 'auto'
            
            # Specification information
            self.spec['goal']     = {1: defSpecBlock(self.partition, a=[-11, -5], b=None, c=[5, 11], d=None)}
            
            self.spec['critical'] = {1: defSpecBlock(self.partition, a=[-9.5,-3.5], b=None, c=[-2,3], d=None),
                                     2: defSpecBlock(self.partition, a=[0, 4.5], b=None, c=[-10, -6], d=None),
                                     3: defSpecBlock(self.partition, a=[-1.2, 5], b=None, c=[-1.5, 3], d=None),
                                     4: defSpecBlock(self.partition, a=[1.0, 4], b=None, c=[8, 11], d=None)}
            
        elif self.modelDim == 3:
            
            self.adaptive = {'rates': [],
                'target_points': np.array([])
                }
            
            # Authority limit for the control u, both positive and negative
            self.control['limits']['uMin'] = [-4, -4, -4]
            self.control['limits']['uMax'] = [4, 4, 4]
            
            # Partition size
            self.partition['nrPerDim']  = [7, 5, 7, 5, 7, 5]
            self.partition['width']     = [2, 1.5, 2, 1.5, 2, 1.5]
            self.partition['origin']    = [0, 0, 0, 0, 0, 0]
            
            # Number of actions per dimension (if 'auto', then equal to nr of regions)
            self.targets['nrPerDim']    = 'auto'
            self.targets['domain']      = 'auto'
            
            # Specification information
            self.spec['goal'] = setStateBlock(self.partition, a=[4,6], b=[None], c=[4,6], d=[None], e=[4,6], f=[None])
            
            self.spec['critical']   = np.vstack((
                setStateBlock(self.partition, a=[-2,0], b=[None], c=[2,4,6], d=[None], e=[None], f=[None]),
                setStateBlock(self.partition, a=[-6,-4], b=[None], c=[4,6], d=[None], e=[4,6], f=[None]),
                setStateBlock(self.partition, a=[-2,0], b=[None], c=[-6,-4,-2,0], d=[None], e=[-6,-4], f=[None]),
                setStateBlock(self.partition, a=[2,4,6], b=[None], c=[-6,-4], d=[None], e=[-6,-4], f=[None]),
                setStateBlock(self.partition, a=[-2,0], b=[None], c=[-6,-4], d=[None], e=[-2,0], f=[None])
                ))
        
        else:
            print('No valid dimension for the drone model was provided')
            sys.exit()
        
        # Step-bound on property
        self.endTime = 16

    def setModel(self, observer):
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
        self.LTI = {}
        
        # Discretization step size
        self.LTI['tau'] = 1.0
        
        self.LTI['noise'] = dict()
        
        # State transition matrix
        Ablock = np.array([[1, self.LTI['tau']],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[self.LTI['tau']**2/2],
                           [self.LTI['tau']]])
        
        if self.modelDim==3:
            self.LTI['A']  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.LTI['B']  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)
            
            # Disturbance matrix
            self.LTI['Q']  = np.array([[0],[0],[0],[0],[0],[0]])
            
            if observer:
                # Observation matrix
                self.LTI['C']          = np.array([[1,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,1,0]])
                self.LTI['r']          = len(self.LTI['C'])
                
                self.LTI['noise']['v_cov'] = np.eye(np.size(self.LTI['C'],0))*0.15
                
            self.filter = {'cov0': np.diag([1, .01, 1, .01, 1, .01])}
                
        else:
            self.LTI['A']  = scipy.linalg.block_diag(Ablock, Ablock)
            self.LTI['B']  = scipy.linalg.block_diag(Bblock, Bblock)
        
            # Disturbance matrix
            self.LTI['Q']  = np.array([[0],[0],[0],[0]])
        
            # Covariance of the process noise
            self.LTI['noise']['w_cov'] = np.diag([0.10, 0.02, 0.10, 0.02])
        
            if observer:
                # Observation matrix
                self.LTI['C']          = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
                self.LTI['r']          = len(self.LTI['C'])
                
                self.LTI['noise']['v_cov'] = np.eye(np.size(self.LTI['C'],0))*0.1

                self.filter = {'cov0': np.diag([4, .01, 4, .01])}
            
        # Determine system dimensions
        self.LTI['n'] = np.size(self.LTI['A'],1)
        self.LTI['p'] = np.size(self.LTI['B'],1)
   
    def setTurbulenceNoise(self, N):
        '''
        Set the turbulence noise samples for N samples

        Parameters
        ----------
        N : int
            Number of samples used.

        Returns
        -------
        None.

        '''
        
        samples = np.genfromtxt('input/TurbulenceNoise_N=1000.csv', delimiter=',')
        
        self.LTI['noise']['samples'] = samples
        
class building_2room(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the 2-zone building automation system (BAS) model class,
        which corresponds to the BAS benchmark in the paper.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Load building automation system (BAS) parameters
        import core.BAS.parameters as BAS_class
        self.BAS = BAS_class.parameters()
        
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.base_delta = 1
        
        self.adaptive = {'rates': [],
                'target_points': np.array([])
                }
        
        # Shortcut to boiler temperature        
        T_boiler = self.BAS.Boiler['Tswbss']
        
        # Authority limit for the control u, both positive and negative
        self.control['limits']['uMin'] = [14, 14, T_boiler-10, T_boiler-10]
        self.control['limits']['uMax'] = [26, 26, T_boiler+10, T_boiler+10]
            
        # Partition size
        self.partition['nrPerDim']  = [11,11,7,7] #[21,21,9,9]
        self.partition['width']     = [0.2, 0.2, 0.2, 0.2]
        self.partition['origin']    = [20, 20, 38.3, 38.3]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.targets['nrPerDim']    = 'auto'
        self.targets['domain']      = 'auto'
        
        # Specification information
        self.spec['goal'] = {1: defSpecBlock(self.partition, 
                                             a=[19.5, 20.5], 
                                             b=[19.5, 20.5], 
                                             c = None,
                                             d = None)}
        
        self.spec['critical'] = {}
        
        # Step-bound on property
        self.endTime = 32

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
        self.LTI = {}
        
        # Discretization step size
        self.LTI['tau'] = 15 # NOTE: in minutes for BAS!
        
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
        
        self.LTI['noise'] = dict()
        
        if observer:
            # Observation matrix
            self.LTI['C']          = np.array([[1,0,0,0]]) #np.array([[1,0,0,0],[0,1,0,0]]) #np.eye(4) #np.array([[1, 0]])
            self.LTI['r']          = len(self.LTI['C'])
            
            self.LTI['noise']['v_cov'] = np.eye(np.size(self.LTI['C'],0))*0.2
            
            self.filter = {'cov0': np.diag([0.05, 0.05, 0.05, 0.05])}
        
        # Discretize model with respect to time
        # self.LTI['A'], self.LTI['B'], self.LTI['Q'] = discretizeGearsMethod(A_cont, B_cont, W_cont, self.LTI['tau'])
        
        self.LTI['A'] = np.eye(4) + self.LTI['tau']*A_cont
        self.LTI['B'] = B_cont*self.LTI['tau']
        self.LTI['Q'] = W_cont*self.LTI['tau']
        
        # Determine system dimensions
        self.LTI['n'] = np.size(self.LTI['A'],1)
        self.LTI['p'] = np.size(self.LTI['B'],1)
        
        self.LTI['noise']['w_cov'] = 0.05*np.diag([0.2, 0.2, 0.2, 0.2])
                
        self.LTI['A_cont'] = A_cont
        self.LTI['B_cont'] = B_cont
        self.LTI['Q_cont'] = W_cont
        
class building_1room(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the 1-zone building automation system (BAS) model class.
        Note that this is a downscaled version of the 2-zone model above.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.base_delta = 1
        
        self.adaptive = {'rates': [],
                'target_points': np.array([])
                }
        
        # Let the user make a choice for the model dimension
        _, gridType  = ui.user_choice('grid size',['19x20','40x40'])
        
        # Authority limit for the control u, both positive and negative
        self.control['limits']['uMin'] = [14, -10]
        self.control['limits']['uMax'] = [28, 10]
            
        if gridType == 0:
            nrPerDim = [19, 20]
            width = [0.2, 0.2]
            goal = [20.6,21.4]#[20.9, 21.1]
        else:
            nrPerDim = [40, 40]
            width = [0.1, 0.1]
            goal = [20.6, 21.4]
        
        # Partition size
        self.partition['nrPerDim']  = nrPerDim
        self.partition['width']     = width
        self.partition['origin']    = [21, 38]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.targets['nrPerDim']    = 'auto'
        self.targets['domain']      = 'auto'
        
        # Specification information
        self.spec['goal'] = {1: defSpecBlock(self.partition, a=goal, b=None)}
        
        self.spec['critical'] = {}
        
        # Step-bound on property
        self.endTime = 16#64

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        
        import core.BAS.parameters as BAS_class
        
        self.LTI = {}
        
        # Discretization step size
        self.LTI['tau'] = 15 # NOTE: in minutes for BAS!
        
        BAS = BAS_class.parameters()
        
        # Steady state values
        Tswb    = BAS.Boiler['Tswbss'] - 20
        Twss    = BAS.Zone1['Twss']
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
        
        self.LTI['noise'] = dict()
        
        if observer:
            # Observation matrix
            self.LTI['C']          = np.eye(2) #np.array([[1, 0]])
            self.LTI['r']          = len(self.LTI['C'])
            
            self.LTI['noise']['v_cov'] = np.eye(np.size(self.LTI['C'],0))*0.1
            
            self.filter = {'cov0': np.diag([0.5, 0.5])}
        
        # Discretize model with respect to time
        # self.LTI['A'], self.LTI['B'], self.LTI['Q'] = discretizeGearsMethod(A_cont, B_cont, W_cont, self.LTI['tau'])
        
        self.LTI['A'] = np.eye(2) + self.LTI['tau']*A_cont
        self.LTI['B'] = B_cont*self.LTI['tau']
        self.LTI['Q'] = W_cont*self.LTI['tau']
        
        # Determine system dimensions
        self.LTI['n'] = np.size(self.LTI['A'],1)
        self.LTI['p'] = np.size(self.LTI['B'],1)
        
        self.LTI['noise']['w_cov'] = np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ])

class shuttle(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the spaceshuttle rendezvous model class.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.base_delta = 2
        
        self.adaptive = {'rates': [],
                'target_points': np.array([])
                }
        
        # Authority limit for the control u, both positive and negative
        self.control['limits']['uMin'] = [-0.1, -0.1]
        self.control['limits']['uMax'] = [0.1, 0.1]
        
        # Partition size
        self.partition['nrPerDim']  = [20, 10, 4, 4]
        self.partition['width']     = [0.1, 0.1, 0.01, 0.01]
        self.partition['origin']    = [0, -0.5, 0.01, 0.01]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.targets['nrPerDim']    = 'auto'
        self.targets['domain']      = 'auto'
        
        # Specification information
        self.spec['goal']     = {1: defSpecBlock(self.partition, a=[-0.2, 0.2], b=[-0.2, 0], c=None, d=None)}
        
        self.spec['critical'] = {1: defSpecBlock(self.partition, a=[-1, -0.2], b=[-0.3, 0], c=None, d=None),
                                 #2: defSpecBlock(self.partition, a=[-1, -0.2], b=[-0.3, -0.2], c=None, d=None),
                                 2: defSpecBlock(self.partition, a=[-1, -0.3], b=[-0.4, -0.3], c=None, d=None),
                                 3: defSpecBlock(self.partition, a=[-1, -0.4], b=[-0.5, -0.4], c=None, d=None),
                                 4: defSpecBlock(self.partition, a=[-1, -0.5], b=[-0.6, -0.5], c=None, d=None),
                                 5: defSpecBlock(self.partition, a=[-1, -0.6], b=[-0.7, -0.6], c=None, d=None),
                                 6: defSpecBlock(self.partition, a=[-1, -0.7], b=[-0.8, -0.7], c=None, d=None),
                                 7: defSpecBlock(self.partition, a=[-1, -0.8], b=[-0.9, -0.8], c=None, d=None),
                                 8: defSpecBlock(self.partition, a=[-1, -0.9], b=[-1.0, -0.9], c=None, d=None),
                                 9: defSpecBlock(self.partition, a=[0.2, 1], b=[-0.3, 0], c=None, d=None),
                                  #11: defSpecBlock(self.partition, a=[0.2, 1], b=[-0.3, -0.2], c=None, d=None),
                                 10: defSpecBlock(self.partition, a=[0.3, 1], b=[-0.4, -0.3], c=None, d=None),
                                 11: defSpecBlock(self.partition, a=[0.4, 1], b=[-0.5, -0.4], c=None, d=None),
                                 12: defSpecBlock(self.partition, a=[0.5, 1], b=[-0.6, -0.5], c=None, d=None),
                                 13: defSpecBlock(self.partition, a=[0.6, 1], b=[-0.7, -0.6], c=None, d=None),
                                 14: defSpecBlock(self.partition, a=[0.7, 1], b=[-0.8, -0.7], c=None, d=None),
                                 15: defSpecBlock(self.partition, a=[0.8, 1], b=[-0.9, -0.8], c=None, d=None),
                                 16: defSpecBlock(self.partition, a=[0.9, 1], b=[-1.0, -0.9], c=None, d=None),}
        
        # Step-bound on property
        self.endTime = 16
        
        self.modelDim = 2

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        self.LTI = {}
        
        # Discretization step size
        self.LTI['tau'] = .7
        
        # Defining Deterministic Model corresponding matrices
        self.LTI['A'] = np.array([[1.000631, 0, 19.9986, 0.410039],
                            [8.62e-6, 1, -0.41004, 19.9944],
                            [6.30e-05, 0, 0.99979, 0.041002],
                            [-1.29e-6, 0, -0.041, 0.999159]])
        
        self.LTI['A'] = (self.LTI['A'] - np.eye(4)) / 2 + np.eye(4)
        
        self.LTI['B'] = np.array([[0.666643, 0.009112],
                           [-0.00911, 0.666573],
                           [0.66662, 0.001367],
                           [-0.00137, 0.66648]]) / 2
        
        self.LTI['Q'] = np.zeros((4,1))
        
        # Determine system dimensions
        self.LTI['n'] = np.size(self.LTI['A'],1)
        self.LTI['p'] = np.size(self.LTI['B'],1)

        self.LTI['noise'] = dict()
        self.LTI['noise']['w_cov'] = 10*np.diag([ 1e-4, 1e-4, 5e-8, 5e-8 ])
        
        if observer:
            # Observation matrix
            self.LTI['C']          = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            self.LTI['r']          = len(self.LTI['C'])
            
            self.LTI['noise']['v_cov'] = np.eye(np.size(self.LTI['C'],0))*0.000001
            
            self.filter = {'cov0': 0.0001*np.diag([.001, .001, .0001, .0001])}
               
class anaesthesia(master.LTI_master):
    
    def __init__(self):
        '''
        Initialize the sanaesthesia delivery model class.

        Returns
        -------
        None.

        '''
        
        # Initialize superclass
        master.LTI_master.__init__(self)
        
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.base_delta = 3
        
        self.adaptive = {'rates': [],
                'target_points': np.array([])
                }
        
        # Authority limit for the control u, both positive and negative
        self.control['limits']['uMin'] = [0]
        self.control['limits']['uMax'] = [7]
        
        # Partition size
        self.partition['nrPerDim']  = [11, 11, 11]
        self.partition['width']     = [.01, .01, .01]
        self.partition['origin']    = [5, 5, 5]
        
        # Number of actions per dimension (if 'auto', then equal to nr of regions)
        self.targets['nrPerDim']    = 'auto'
        self.targets['domain']      = 'auto'
        
        # Specification information
        self.spec['goal']     = {1: defSpecBlock(self.partition, a=[4.9, 5.1], b=[4.9, 5.1], c=[4.9, 5.1])}
        self.spec['critical'] = {}
        
        # Step-bound on property
        self.endTime = 16
        
        self.modelDim = 3

    def setModel(self, observer):           
        '''
        Set linear dynamical system.

        Parameters
        ----------
        observer : Boolean
            If True, an observer is created for the model.

        Returns
        -------
        None.

        '''
        self.LTI = {}
        
        # Discretization step size
        self.LTI['tau'] = 1
        
        '''
        k10 = 0.4436
        k12 = 0.1140
        k13 = 0.0419
        k21 = 0.550
        k31 = 0.0033
        V1  = 16.044
        
        # Defining Deterministic Model corresponding matrices
        self.LTI['A'] = np.array([
                            [-(k10+k12+k13),    k12,    k13],
                            [k21,               -k21,   0],
                            [k31,               0,      -k31],
                            ])
        
        self.LTI['B'] = np.array([
                           [1/V1],
                           [0],
                           [0]
                           ]) / 2
        '''
        
        # Defining Deterministic Model corresponding matrices
        self.LTI['A'] = np.array([
                            [0.8192,    0.3412,    0.1265],
                            [0.1646,   0.9822,     0.0001],
                            [0.09,    0.00002,    0.9989],
                            ])
        
        self.LTI['B'] = np.array([
                           [0.01883],
                           [0.02],
                           [0.001]
                           ])
        
        self.LTI['Q'] = np.zeros((3,1))
        
        # Determine system dimensions
        self.LTI['n'] = np.size(self.LTI['A'],1)
        self.LTI['p'] = np.size(self.LTI['B'],1)

        self.LTI['noise'] = dict()
        self.LTI['noise']['w_cov'] = 10**-3 * np.eye(3)
        
        if observer:
            # Observation matrix
            self.LTI['C']          = np.eye(3)
            self.LTI['r']          = len(self.LTI['C'])
            
            self.LTI['noise']['v_cov'] = np.eye(np.size(self.LTI['C'],0))*0.0001
            
            self.filter = {'cov0': np.diag([.001, .001, .001])}
               