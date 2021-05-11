import numpy as np
import itertools

from .commons import table, is_invertible
                        
import core.modelDefinitions as models

from scipy.stats import mvn
from scipy.spatial import Delaunay

def in_hull(p, hull):
    '''
    Test if points in `p` are in `hull`.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    '''
    
    if not isinstance(hull,Delaunay):
        print(' -- Creating hull...')
        hull = Delaunay(hull, qhull_options='QJ')

    boolArray = hull.find_simplex(p) >= 0

    return boolArray

def computeGaussianProbabilities(partitions, mu, sigma):
    '''
    Calculate the transition probabilities to reach any region in `partitions`
    when the posterior belief has mean `mu` and covariance `sigma`.

    Parameters
    ----------
    partitions : dict
        Dictionary containing the info regarding partisions.
    mu : ndarray
        vector of length n for the mean of the posterior belief.
    S : np.array
        n by n matrix of the covariance of the posterior belief.

    Returns
    -------
    list
        List of probabilities to reach any state in the list of partitions.

    '''

    return [ mvn.mvnun(partitions[i]['low'], 
                       partitions[i]['upp'], mu, sigma)[0] 
            for i in range(len(partitions)) ]

def computeRegionCenters(points, partition):
    '''
    Function to compute to which region (center) a list of points belong
    '''
    
    # Retreive partition parameters
    region_width = np.array(partition['width'])
    region_nrPerDim = partition['nrPerDim']
    dim = len(region_width)
    
    # Boolean list per dimension if it has a region with the origin as center
    originCentered = np.zeros(dim)
    for q in range(dim):
        if region_nrPerDim[q]/2 != int(region_nrPerDim[q]/2):
            originCentered[q] = True

    # Initialize centers array
    centers = np.zeros(np.shape(points)) 
    
    # Shift the points to account for a non-zero origin
    originShift = np.array(partition['origin'] )
    pointsShift = points - originShift
    
    for q in range(dim):
        # Compute the center coordinates of every shifted point
        if originCentered[q]:
            centers[:,q] = ((pointsShift[:,q]+0.5*region_width[q]) // region_width[q]) * region_width[q]
        else:
            centers[q] = (pointsShift[:,q] // region_width[q]) * region_width[q] + 0.5*region_width[q]
    
    # Add the origin again to obtain the absolute center coordinates
    return centers + originShift

def computeScenarioBounds(setup, partition, abstr, samples):
    
    # Number of decision variables always equal to one
    d = 1
    Nsamples = setup.scenarios['samples']
    beta = setup.scenarios['confidence']
    
    # Initialize vectors for probability bounds
    probability_low = np.zeros(abstr['nr_regions'])
    probability_upp = np.zeros(abstr['nr_regions'])
    
    # Initialize counts array
    counts = np.zeros(abstr['nr_regions'])
    
    centers = computeRegionCenters(samples, partition)
    
    for s in range(Nsamples):
        
        key = tuple(centers[s])
        
        if key in abstr['allCenters']:
            counts[abstr['allCenters'][ key ]] += 1

    nonEmpty = counts > 0
    
    # Count number of samples not in any region (i.e. in absorbing state)
    k = Nsamples - sum(counts) + d

    key_lb = tuple( [Nsamples, k, beta] )
    key_ub = tuple( [Nsamples, k-1, beta] ) 
    
    deadlock_low = abstr['memory'][key_ub][0]
    if k > Nsamples:
        deadlock_upp = 1
    else:
        deadlock_upp = abstr['memory'][key_lb][1]

    # Enumerate over all the non-zero bins
    for i, count in zip(np.arange(abstr['nr_regions'])[nonEmpty], counts[nonEmpty]):
        
        if count > 0:
        
            k = Nsamples - count + d
            
            key_lb = tuple( [Nsamples, k,   beta] )
            key_ub = tuple( [Nsamples, k-1, beta] )
            
            if k > Nsamples:
                probability_low[i] = 0                
            else:
                probability_low[i] = 1 - abstr['memory'][key_lb][1]
            probability_upp[i] = 1 - abstr['memory'][key_ub][0]
    
    returnDict = {
        'lb': probability_low,
        'ub': probability_upp,
        'deadlock_lb': deadlock_low,
        'deadlock_ub': deadlock_upp,
        'approx': probability_low
    }
    
    return abstr['memory'], returnDict

def kalmanFilter(model, cov0):    
    '''
    For a given model in `model` and prior belief covariance `cov0`, perform
    The Kalman filter steps related to the covariance (mean is not needed).

    Parameters
    ----------
    model : dict
        Main dictionary of the LTI system model.
    cov0 : ndarray
        n by n matrix of the covariance of the prior belief.

    Returns
    -------
    cov_pred : ndarray
        Predicted covariance (after the prediction step).
    K_gain : ndarray
        Optimal Kalman gain.
    cov_tilde : ndarray
        Covariance of the mean of the posterior belief.
    cov : ndarray
        Covariance of the posterior belief.
    cov_tilde_measure : float
        Measure on the covariance of the mean of the posterior belief.

    '''
    
    # Predicted covariance of the belief
    cov_pred = model.A @ cov0 @ model.A.T + model.noise['w_cov']
    
    # If matrix is invertible
    if is_invertible(model.C @ cov_pred @ model.C.T + model.noise['v_cov']):
        # Determine optimal Kalman gain
        K_gain   = cov_pred @ model.C.T @ \
            np.linalg.inv(model.C @ cov_pred @ model.C.T + model.noise['v_cov'])
        
    else:
        # If not invertible, set Kalman gain to zero
        K_gain   = np.zeros((model.n, model.r))
        
    # Covariance of the expected mean of the future belief
    cov_tilde = K_gain @ \
    (model.C @ cov_pred @ model.C.T + model.noise['v_cov']) @ K_gain.T
        
    # Covariance of the future belief
    cov = (np.eye(model.n) - K_gain @ model.C) @ cov_pred
    
    # Calculate measure on the covariance matrix of the mean of the future belief
    cov_tilde_measure = covarianceEllipseSize( cov_tilde )
    
    return cov_pred, K_gain, cov_tilde, cov, cov_tilde_measure

def covarianceEllipseSize(cov):
    '''
    Computes the maximum radius of the ellipse that is associated with the
    covariance matrix `cov`.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix.

    Returns
    -------
    ellipse_size : float
        Maximum radius of the ellipse associated with the covariance matrix.

    '''
    
    # Determine the largest size parameter of the rotated ellipse associated
    # with the given covariance matrix
    
    # Determine eigenvectors and values
    eigenval, eigenvec = np.linalg.eig( cov )
    
    # Retreive the largest value
    max_eigenval = eigenval[np.argmax(eigenval)]
    
    # Determine largest width parameter of ellipse
    ellipse_size = np.sqrt( max_eigenval )
    
    return ellipse_size

def definePartitions(dim, nrPerDim, regionWidth, origin, onlyCenter=False):
    '''
    Define the partitions object `partitions` based on given settings.

    Parameters
    ----------
    dim : int
        Dimension of the state (of the LTI system).
    off_number : list
        Number of regions to be defined in every dimension from the center.
    width : list
        Width of a region
    origin : list
        Coordinates of the origin of the continuous state space.

    Returns
    -------
    partitions : dict
        Dictionary containing the info regarding partisions.
    goal_index : int
        Index of the goal region of the partition.

    '''
    
    regionWidth = np.array(regionWidth)
    origin      = np.array(origin)
    
    elemVector   = dict()
    for i in range(dim):
        
        elemVector[i] = np.linspace(-(nrPerDim[i]-1)/2, 
                                     (nrPerDim[i]-1)/2,
                                     int(nrPerDim[i]))
        
    widthArrays = [[x*regionWidth[i] for x in elemVector[i]] 
                                              for i in range(dim)]
    
    if onlyCenter:        
        partitions = np.array(list(itertools.product(*widthArrays))) + origin
        
    else:
        partitions = dict()
        
        for i,elements in enumerate(itertools.product(*widthArrays)):
            partitions[i] = dict()
            
            center = np.array(elements) + origin
            
            partitions[i]['center'] = center
            partitions[i]['low'] = center - regionWidth/2
            partitions[i]['upp'] = center + regionWidth/2
    
    return partitions

def makeModelFullyActuated(model, manualDimension='auto', observer=True):
    '''
    Given a model in `model`, render it fully actuated.

    Parameters
    ----------
    model : dict
        Main dictionary of the LTI system model.
    manualDimension : int or str, optional
        Desired dimension of the state of the model The default is 'auto'.

    Returns
    -------
    model : dict
        Main dictionary of the LTI system model, which is now fully actuated.

    '''
    
    if manualDimension == 'auto':
        # Determine dimension for actuation transformation
        dim    = int( np.size(model.A,1) / np.size(model.B,1) )
    else:
        # Group a manual number of time steps
        dim    = int( manualDimension )
    
    # Determine fully actuated system matrices and parameters
    A_hat  = np.linalg.matrix_power(model.A, (dim))
    B_hat  = np.concatenate([ np.linalg.matrix_power(model.A, (dim-i)) \
                                      @ model.B for i in range(1,dim+1) ], 1)
    
    w_mean_hat = sum([ np.linalg.matrix_power(model.A, (dim-i)) @ model.noise['w_mean']
                       for i in range(1,dim+1) ])
    
    w_sigma_hat  = sum([ np.array( np.linalg.matrix_power(model.A, (dim-i) ) \
                                      @ model.noise['w_cov'] @ \
                                      np.linalg.matrix_power(model.A.T, (dim-i) ) \
                                    ) for i in range(1,dim+1) ])
    
    # Overwrite original system matrices
    model.A               = A_hat
    model.B               = B_hat
   
    model.noise['w_mean'] = w_mean_hat
    model.noise['w_cov']  = w_sigma_hat
    
    # Redefine sampling time of model
    model.tau             *= (dim+1)
    
    return model

def defineDistances(partitions, base_region):
    '''
    Define the mutual difference in all dimensions between any two target points.

    Parameters
    ----------
    partitions : dict
        Dictionary containing the info regarding partisions.
    base_region : int
        Index of the base region to which we want to measure.

    Returns
    -------
    distance_list : list
        List of differences in coordinates to the `base_region`.

    '''
    
    # Define center of the base region as point A
    pointA = np.array(partitions[base_region]['center'])
    
    # For the center of every other region, compute difference to point A
    distance_list = [ pointA - region['center']
                     for region in partitions.values()]
        
    return distance_list 

def valueIteration(setup, mdp):
    '''
    Solve the MDP defined in Python internally via Value iteration.

    Parameters
    ----------
    setup : dict
        Setup dictionary.
    mdp : mdp
        MDP object.

    Returns
    -------
    ndarray
        N x |P| array containing the actions of the optimal policy.
    ndarray
        N x |P| array containing the delta values of the optimal policy.
    ndarray
        N x |P| array containing the optimal reachability probability.

    '''
    
    # Retreive time horizon of the MDP
    N = mdp.N
    
    # Determine last time step
    reverse_time_steps = np.arange(N)[::-1]

    # Define zero vectors for results
    reward_vector = np.zeros(mdp.nr_states + max(setup.deltas)*mdp.nr_regions)
    delta_vector  = [None for i in range(mdp.nr_states)]
    policy_vector = [None for i in range(mdp.nr_states)]
    
    # For every "good" end state, set the reachability probability to one, and
    # for every "bad" state, set it to zero
    reward_vector[mdp.goodStates] = 1
    reward_vector[mdp.badStates] = 0

    # Column widths for tabular prints
    col_width = [8,46]
    tab = table(col_width)

    # Print header row
    tab.print_row(['TIME','STATUS'], head=True)

    # Loop over reserved time steps within optimization horizon
    for k in reverse_time_steps:
        
        # Print normal table row
        tab.print_row([k,'Solve previous time step'])
        
        # Loop over each of the current states
        for s,current_actions in mdp.transitions['succ'][k].items():

            if s in mdp.goodStates:
                # If state is target state, set probability to one
                reward_vector[s] = 1
                delta_vector[s]  = -1
                policy_vector[s] = -1
                
            elif s in mdp.badStates:
                
                # If state is critical state, set probability to zero
                reward_vector[s] = 0
                delta_vector[s]  = -1
                policy_vector[s] = -1
                
            else:

                # Empty vector for probabilities
                if len(current_actions) > 0:
                    
                    current_rewards = [mdp.transitions['prob'][k][s][key] @ 
                                       reward_vector[ mdp.transitions['succ'][k][s][key] ] for
                                       key in current_actions.keys()]
                    current_actions_keys = [key for key in current_actions.keys()]
                        
                    # Determine the optimal action, by maximizing the probability
                    reward_vector[s] = np.max(current_rewards)
                    
                    # Avoid using argmax, since we want to return a list of actions
                    # if the optimal policy is not unique
                    delta_vector[s] = current_actions_keys[ np.argmax(current_rewards) ][0]
                    policy_vector[s] = current_actions_keys[ np.argmax(current_rewards) ][1]
                    
                else:
                    
                    # No policy known
                    reward_vector[s] = 0
                    delta_vector[s]  = -1
                    policy_vector[s] = -1
           
    solution = {
        'optimal_policy': policy_vector[:mdp.nr_states],
        'optimal_delta': delta_vector[:mdp.nr_states],
        'optimal_reward': reward_vector[:mdp.nr_states],
        }
           
    return solution